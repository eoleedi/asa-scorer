from graphviz import Digraph


def generate_architecture_diagram():
    dot = Digraph(name="FDMPAScorer_Updated", format="png")

    dot.attr(
        rankdir="LR",
        splines="ortho",
        nodesep="0.8",
        ranksep="1.0",
        compound="true",
    )

    dot.attr("graph", center="true", pad="0.3")
    dot.attr(
        "node",
        shape="box",
        style="filled",
        fontname="Arial",
        margin="0.18",
        fontsize="16",
    )
    dot.attr("edge", fontname="Arial", fontsize="10")

    # ==========================================================
    # BLOCK 1: INPUT & PROCESSING
    # ==========================================================
    with dot.subgraph(name="cluster_input") as c1:
        c1.attr(
            label="Input & Processing",
            style="dashed",
            color="blue",
            bgcolor="#e8f4f8",
            fontname="Arial-Bold",
            fontsize="16",
        )

        with c1.subgraph() as s:
            s.attr(rank="same")
            # Top anchors keep raw audio centered over the lower row.
            s.node("L0", "", width="1.8", style="invis", fixedsize="true")
            s.node(
                "Audio",
                "Raw Audio Waveform\n(L2 Learner Speech)",
                fillcolor="white",
                width="3.2",
                height="0.8",
                fixedsize="true",
            )
            s.node("R0", "", width="1.8", style="invis", fixedsize="true")
            s.edge("L0", "Audio", style="invis")
            s.edge("Audio", "R0", style="invis")

        with c1.subgraph() as s:
            s.attr(rank="same")
            # Invisible Left Anchor (Matches width of FVQ_pro)
            s.node("L1", "", width="1.8", style="invis", fixedsize="true")

            s.node(
                "HuBERT",
                "HuBERT / Emotion2Vec Extractor\nSSL Features (T × 768)",
                fillcolor="#cce5ff",
                width="3.6",
                height="0.9",
                fixedsize="true",
            )
            s.node(
                "ProMoNet",
                "ProMoNet Extractor\nHC Features (T × 50)",
                fillcolor="#cce5ff",
                width="3.6",
                height="0.9",
                fixedsize="true",
            )

            # Invisible Right Anchor (Matches width of Z_rhy)
            s.node("R1", "", width="1.8", style="invis", fixedsize="true")

            # Flex horizontal sequence
            s.edge("L1", "HuBERT", style="invis")
            s.edge("HuBERT", "ProMoNet", style="invis")
            s.edge("ProMoNet", "R1", style="invis")

        # Align upper and lower boundaries so Audio stays horizontally centered.
        c1.edge("L0", "L1", style="invis", weight="1000")
        c1.edge("R0", "R1", style="invis", weight="1000")

        c1.edge("Audio", "HuBERT")
        c1.edge("Audio", "ProMoNet")

    # ==========================================================
    # BLOCK 2: FDMPAScorer BOTTLENECK (3 Identical Areas)
    # ==========================================================
    with dot.subgraph(name="cluster_bottleneck") as c2:
        c2.attr(
            label="FDMPAScorer Bottleneck",
            style="dashed",
            color="green",
            bgcolor="#e8f5e9",
            fontname="Arial-Bold",
            fontsize="16",
        )

        with c2.subgraph() as s:
            s.attr(rank="same")

            # All 6 nodes are strictly the same width (1.8) to make the 3 feature areas equal
            s.node(
                "FVQ_pro",
                "Prominence FVQ\nSSL Latent",
                fillcolor="#d4edda",
                width="1.8",
                height="0.8",
                fixedsize="true",
            )
            s.node(
                "Z_pro",
                "Z_pro\nLoudness",
                fillcolor="#fff3cd",
                color="#fbc02d",
                penwidth="2",
                width="1.8",
                height="0.8",
                fixedsize="true",
            )

            s.node(
                "FVQ_int",
                "Intonation FVQ\nSSL Latent",
                fillcolor="#d4edda",
                width="1.8",
                height="0.8",
                fixedsize="true",
            )
            s.node(
                "Z_int",
                "Z_int\nPitch / Periodicity",
                fillcolor="#fff3cd",
                color="#fbc02d",
                penwidth="2",
                width="1.8",
                height="0.8",
                fixedsize="true",
            )

            s.node(
                "FVQ_rhy",
                "Rhythm FVQ\nSSL Latent",
                fillcolor="#d4edda",
                width="1.8",
                height="0.8",
                fixedsize="true",
            )
            s.node(
                "Z_rhy",
                "Z_rhy\nPPG",
                fillcolor="#fff3cd",
                color="#fbc02d",
                penwidth="2",
                width="1.8",
                height="0.8",
                fixedsize="true",
            )

            # Rigid invisible horizontal backbone
            s.edge("FVQ_pro", "Z_pro", style="invis", weight="10")
            s.edge("Z_pro", "FVQ_int", style="invis", weight="10")
            s.edge("FVQ_int", "Z_int", style="invis", weight="10")
            s.edge("Z_int", "FVQ_rhy", style="invis", weight="10")
            s.edge("FVQ_rhy", "Z_rhy", style="invis", weight="10")

        # Visible MINE Loss Edges (Floating so they don't break layout)
        c2.edge(
            "FVQ_pro",
            "Z_pro",
            xlabel="MINE Loss",
            constraint="false",
            color="gray40",
            fontcolor="gray30",
            penwidth="2",
            minlen="2",
        )
        c2.edge(
            "FVQ_int",
            "Z_int",
            xlabel="MINE Loss",
            constraint="false",
            color="gray40",
            fontcolor="gray30",
            penwidth="2",
            minlen="2",
        )
        c2.edge(
            "FVQ_rhy",
            "Z_rhy",
            xlabel="MINE Loss",
            constraint="false",
            color="gray40",
            fontcolor="gray30",
            penwidth="2",
            minlen="2",
        )

    # ==========================================================
    # BLOCK 3: HIERARCHICAL POOLING & OUTPUT
    # ==========================================================
    with dot.subgraph(name="cluster_pooling") as c3:
        c3.attr(
            label="Hierarchical Pooling & Output",
            style="dashed",
            color="purple",
            bgcolor="#f3e5f5",
            fontname="Arial-Bold",
            fontsize="16",
        )

        with c3.subgraph() as s:
            s.attr(rank="same")
            # Invisible Left Anchor
            s.node("L3", "", width="1.8", style="invis", fixedsize="true")

            # Uniform pooling nodes
            s.node(
                "Pool_pro",
                "AttentiveStatsPooling",
                fillcolor="#e2d9f3",
                width="2.6",
                height="0.7",
                fixedsize="true",
            )
            s.node(
                "Pool_int",
                "AttentiveStatsPooling",
                fillcolor="#e2d9f3",
                width="2.6",
                height="0.7",
                fixedsize="true",
            )
            s.node(
                "Pool_rhy",
                "AttentiveStatsPooling",
                fillcolor="#e2d9f3",
                width="2.6",
                height="0.7",
                fixedsize="true",
            )

            # Invisible Right Anchor
            s.node("R3", "", width="1.8", style="invis", fixedsize="true")

            # Flex horizontal sequence
            s.edge("L3", "Pool_pro", style="invis")
            s.edge("Pool_pro", "Pool_int", style="invis")
            s.edge("Pool_int", "Pool_rhy", style="invis")
            s.edge("Pool_rhy", "R3", style="invis")

        c3.node(
            "Concat",
            "Global Feature Fusion\nConcat(Mean, Std)",
            fillcolor="#d1b3ff",
            width="3.4",
            height="0.8",
            fixedsize="true",
        )
        c3.node(
            "Head",
            "Final Linear Head",
            fillcolor="#ffcccc",
            width="2.6",
            height="0.7",
            fixedsize="true",
        )
        c3.node(
            "Score",
            "Prosodic Score",
            shape="ellipse",
            fillcolor="#ffcccc",
            style="filled,bold",
            color="red",
            width="2.4",
            height="0.7",
            fixedsize="true",
        )

        c3.edge("Pool_pro", "Concat")
        c3.edge("Pool_int", "Concat")
        c3.edge("Pool_rhy", "Concat")
        c3.edge("Concat", "Head")
        c3.edge("Head", "Score")

    # ==========================================================
    # STRUCTURAL PILLARS (The Magic Step)
    # ==========================================================
    # We strictly align the invisible anchors to the outer nodes of the bottleneck.
    # This forces all three dashed cluster boxes to be the exact same width.

    # Left Box Boundary
    dot.edge("L1", "FVQ_pro", style="invis", weight="1000")
    dot.edge("FVQ_pro", "L3", style="invis", weight="1000")

    # Right Box Boundary
    dot.edge("R1", "Z_rhy", style="invis", weight="1000")
    dot.edge("Z_rhy", "R3", style="invis", weight="1000")

    # ==========================================================
    # FLOATING WIRING (constraint="false")
    # ==========================================================
    # Because our structural pillars hold the diagram together, all visible
    # cross-block wires can float. They will draw cleanly without shuffling nodes.

    # HuBERT inputs
    dot.edge("HuBERT", "FVQ_pro", constraint="false")
    dot.edge("HuBERT", "FVQ_int", constraint="false")
    dot.edge("HuBERT", "FVQ_rhy", constraint="false")

    # ProMoNet inputs
    dot.edge("ProMoNet", "Z_pro", constraint="false")
    dot.edge("ProMoNet", "Z_int", constraint="false")
    dot.edge("ProMoNet", "Z_rhy", constraint="false")

    # Bottleneck to Pooling
    dot.edge("FVQ_pro", "Pool_pro", constraint="false")
    dot.edge("FVQ_int", "Pool_int", constraint="false")
    dot.edge("FVQ_rhy", "Pool_rhy", constraint="false")

    dot.render("Architecture", view=False, cleanup=True)
    print("Successfully generated Architecture.png")


if __name__ == "__main__":
    generate_architecture_diagram()
