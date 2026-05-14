from graphviz import Digraph


def generate_vq_diagram():
    dot = Digraph(name="FactorizedVectorQuantize", format="png")
    dot.attr(
        rankdir="TB", splines="ortho", nodesep="0.6", ranksep="0.6", compound="true"
    )
    dot.attr("node", shape="box", style="filled", fontname="Arial", margin="0.2")

    # Input Node
    dot.node("Input", "Input Tensor: z\n[B, D, T]", fillcolor="#e3f2fd")

    # Projections
    dot.node("in_proj", "in_proj\nLinear(dim → codebook_dim)", fillcolor="#e8f5e9")
    dot.node("ZE", "Continuous Latent: z_e", fillcolor="#e3f2fd", style="filled,bold")

    dot.edge("Input", "in_proj", label=" transpose(1,2)")
    dot.edge("in_proj", "ZE", label=" transpose(1,2)")

    # Vector Quantization Cluster
    with dot.subgraph(name="cluster_vq") as vq:
        vq.attr(
            label="decode_latents()", style="dashed", color="blue", bgcolor="#f4f8ff"
        )

        vq.node("Norm", "L2 Normalize z_e & codebook", fillcolor="#fff9c4")
        vq.node(
            "Dist",
            "Compute L2 Distance\n(via dot product expansion)",
            fillcolor="#fff9c4",
        )
        vq.node(
            "Argmin",
            "indices = (-dist).max(1)[1]",
            fillcolor="#ffcc80",
            shape="ellipse",
        )
        vq.node(
            "Codebook",
            "Codebook Embeddings\n[codebook_size, codebook_dim]",
            fillcolor="#d1c4e9",
            shape="cylinder",
        )
        vq.node("Decode", "decode_code(indices)", fillcolor="#fff9c4")
        vq.node(
            "ZQ_raw",
            "Discrete Latent: z_q (raw)",
            fillcolor="#e3f2fd",
            style="filled,bold",
        )

        vq.edge("Norm", "Dist")
        vq.edge("Codebook", "Dist")
        vq.edge("Dist", "Argmin")
        vq.edge("Argmin", "Decode")
        vq.edge("Codebook", "Decode", style="dotted")
        vq.edge("Decode", "ZQ_raw")

    dot.edge("ZE", "Norm")

    # Loss Cluster
    with dot.subgraph(name="cluster_loss") as loss:
        loss.attr(label="Training Loss", style="dashed", color="red", bgcolor="#ffebee")
        loss.node(
            "CommitLoss",
            "Commitment Loss\nMSE(z_e, sg(z_q)) * commit",
            fillcolor="#ffcdd2",
        )
        loss.node(
            "CodebookLoss", "Codebook Loss\nMSE(z_q, sg(z_e))", fillcolor="#ffcdd2"
        )
        loss.node("TotalLoss", "commit_loss", fillcolor="#ef9a9a", shape="ellipse")

        loss.edge("CommitLoss", "TotalLoss")
        loss.edge("CodebookLoss", "TotalLoss")

    dot.edge("ZE", "CommitLoss", style="dotted", label=" z_e")
    dot.edge("ZQ_raw", "CommitLoss", style="dotted", label=" sg(z_q)")
    dot.edge("ZQ_raw", "CodebookLoss", style="dotted", label=" z_q")
    dot.edge("ZE", "CodebookLoss", style="dotted", label=" sg(z_e)")

    # Straight-Through Estimator
    dot.node(
        "STE",
        "Straight-Through Estimator\nz_q = z_e + (z_q - z_e).detach()",
        fillcolor="#ffeb3b",
        style="filled,bold",
    )

    # Forward Pass through STE
    dot.edge("ZE", "STE", color="black", penwidth="2")
    dot.edge("ZQ_raw", "STE", color="gray", style="dashed", label=" .detach()")

    # Output Projections
    dot.node("ZQ_ste", "z_q (with gradients)", fillcolor="#e3f2fd", style="filled,bold")
    dot.node("out_proj", "out_proj\nLinear(codebook_dim → dim)", fillcolor="#e8f5e9")

    # Final Outputs
    dot.node("Out_ZQ", "Output: z_q\n[B, D, T]", fillcolor="#c8e6c9", shape="ellipse")
    dot.node("Out_Idx", "Output: indices\n[B, T]", fillcolor="#c8e6c9", shape="ellipse")

    dot.edge("STE", "ZQ_ste")
    dot.edge("ZQ_ste", "out_proj", label=" transpose(1,2)")
    dot.edge("out_proj", "Out_ZQ", label=" transpose(1,2)")
    dot.edge("Argmin", "Out_Idx", constraint="false")

    # Gradient flow annotation (Backprop)
    dot.edge(
        "STE",
        "ZE",
        color="red",
        fontcolor="red",
        style="dashed",
        label=" Gradients bypass argmin",
        constraint="false",
    )

    # Render
    dot.render("FactorizedVectorQuantize", view=False, cleanup=True)
    print("Successfully generated FactorizedVectorQuantize.png")


if __name__ == "__main__":
    generate_vq_diagram()
