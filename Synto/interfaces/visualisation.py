"""
Module containing functions for analysis and visualization of the built search tree
"""

from itertools import count, islice

from CGRtools.containers import MoleculeContainer
from tqdm import tqdm as _tqdm


def tqdm(disable=True, *args, **kwargs):
    """
    It is used to create a progress bar with the option to hide the output.

    :param disable: Determines whether the tqdm progress bar should be disabled or not
    :return: The tqdm object
    """
    # Function to be able to hide Tree tqdm output.
    kwargs.setdefault('disable', disable)
    return _tqdm(disable, *args, **kwargs)


def extract_routes(tree, extended=False):
    """
    The function takes the target target and the dictionary of
    successors and predecessors and returns a list of dictionaries that contain the target target
    and the list of
    successors
    :return: A list of dictionaries. Each dictionary contains a target, a list of children, and a
    boolean indicating whether the target is in building_blocks.
    """
    target = tree.nodes[1].retrons_to_expand[0].molecule
    target_in_stock = tree.nodes[1].curr_retron.is_building_block(tree.building_blocks)
    # Append encoded routes to list
    paths_block = []
    winning_nodes = []
    if extended:
        # Gather paths
        for indice, node in tree.nodes.items():
            if node.is_solved():
                winning_nodes.append(indice)
    else:
        winning_nodes = tree.winning_nodes
    if winning_nodes:
        for winning_node in winning_nodes:
            # Create graph for route
            nodes = tree.path_to_node(winning_node)
            succ = {}
            pred = {}
            for before, after in zip(nodes, nodes[1:]):
                before = before.curr_retron.molecule
                succ[before] = after = [x.molecule for x in after.new_retrons]
                for x in after:
                    pred[x] = before

            paths_block.append(
                {
                    "type": "mol",
                    "smiles": str(target),
                    "in_stock": target_in_stock,
                    "children": [get_child_nodes(tree, target, succ)],
                }
            )
    else:
        paths_block = [
            {
                "type": "mol",
                "smiles": str(target),
                "in_stock": target_in_stock,
                "children": [],
            }
        ]
    return paths_block


def path_graph(tree, node: int) -> str:
    """
    Visualizes reaction path

    :param node: int
    :type node: int
    :return: The SVG string.
    """
    nodes = tree.path_to_node(node)
    # Set up node_id types for different box colors
    for node in nodes:
        for retron in node.new_retrons:
            retron._molecule.meta["status"] = "instock" if retron.is_building_block(
                tree.building_blocks) else "mulecule"
    nodes[0].curr_retron._molecule.meta["status"] = "target"
    # Box colors
    box_colors = {
        "target": "#98EEFF",  # 152, 238, 255
        "mulecule": "#F0AB90",  # 240, 171, 144
        "instock": "#9BFAB3",  # 155, 250, 179
    }

    # first column is target
    # second column are first new retrons_to_expand
    columns = [[nodes[0].curr_retron.molecule], [x.molecule for x in nodes[1].new_retrons], ]
    pred = {x: 0 for x in range(1, len(columns[1]) + 1)}
    cx = [n for n, x in enumerate(nodes[1].new_retrons, 1) if not x.is_building_block(tree.building_blocks)]
    size = len(cx)
    nodes = iter(nodes[2:])
    cy = count(len(columns[1]) + 1)
    while size:
        layer = []
        for s in islice(nodes, size):
            n = cx.pop(0)
            for x in s.new_retrons:
                layer.append(x)
                m = next(cy)
                if not x.is_building_block(tree.building_blocks):
                    cx.append(m)
                pred[m] = n
        size = len(cx)
        columns.append([x.molecule for x in layer])

    columns = [columns[::-1] for columns in columns[::-1]]  # Reverse array to make retrosynthetic graph
    pred = tuple(  # Change dict to tuple to make multiple retrons_to_expand available
        (abs(source - len(pred)), abs(target - len(pred)))
        for target, source in pred.items()
    )

    # now we have columns for visualizing
    # lets start recalculate XY
    x_shift = 0.0
    c_max_x = 0.0
    c_max_y = 0.0
    render = []
    cx = count()
    cy = count()
    arrow_points = {}
    for ms in columns:
        heights = []
        for m in ms:
            m.clean2d()
            # X-shift for target
            min_x = min(x for x, y in m._plane.values()) - x_shift
            min_y = min(y for x, y in m._plane.values())
            m._plane = {n: (x - min_x, y - min_y) for n, (x, y) in m._plane.items()}
            max_x = max(x for x, y in m._plane.values())
            if max_x > c_max_x:
                c_max_x = max_x
            arrow_points[next(cx)] = [x_shift, max_x]
            heights.append(max(y for x, y in m._plane.values()))

        x_shift = c_max_x + 5.0  # between columns gap
        # calculate Y-shift
        y_shift = sum(heights) + 3.0 * (len(heights) - 1)
        if y_shift > c_max_y:
            c_max_y = y_shift
        y_shift /= 2.0
        for m, h in zip(ms, heights):
            m._plane = {n: (x, y - y_shift) for n, (x, y) in m._plane.items()}

            # Calculate coordinates for boxes
            max_x = max(x for x, y in m._plane.values()) + 0.9  # Max x
            min_x = min(x for x, y in m._plane.values()) - 0.6  # Min x
            max_y = -(max(y for x, y in m._plane.values()) + 0.45)  # Max y
            min_y = -(min(y for x, y in m._plane.values()) - 0.45)  # Min y
            x_delta = abs(max_x - min_x)
            y_delta = abs(max_y - min_y)
            box = (
                f'<rect x="{min_x}" y="{max_y}" rx="{y_delta * 0.1}" ry="{y_delta * 0.1}" width="{x_delta}" height="{y_delta}"'
                f' stroke="black" stroke-width=".0025" fill="{box_colors[m.meta["status"]]}" fill-opacity="0.30"/>'
            )
            arrow_points[next(cy)].append(y_shift - h / 2.0)
            y_shift -= h + 3.0
            depicted_molecule = list(m.depict(embedding=True))[:3]
            depicted_molecule.append(box)
            render.append(depicted_molecule)

    # Calculate mid-X coordinate to draw square arrows
    graph = {}
    for s, p in pred:
        try:
            graph[s].append(p)
        except KeyError:
            graph[s] = [p]
    for s, ps in graph.items():
        mid_x = float("-inf")
        for p in ps:
            s_min_x, s_max, s_y = arrow_points[s][:3]  # s
            p_min_x, p_max, p_y = arrow_points[p][:3]  # p
            p_max += 1
            mid = p_max + (s_min_x - p_max) / 3
            if mid > mid_x:
                mid_x = mid
        for p in ps:
            arrow_points[p].append(mid_x)

    config = MoleculeContainer._render_config
    font_size = config["font_size"]
    font125 = 1.25 * font_size
    width = c_max_x + 4.0 * font_size  # 3.0 by default
    height = c_max_y + 3.5 * font_size  # 2.5 by default
    box_y = height / 2.0
    svg = [
        f'<svg width="{0.6 * width:.2f}cm" height="{0.6 * height:.2f}cm" '
        f'viewBox="{-font125:.2f} {-box_y:.2f} {width:.2f} '
        f'{height:.2f}" xmlns="http://www.w3.org/2000/svg" version="1.1">',
        '  <defs>\n    <marker id="arrow" markerWidth="10" markerHeight="10" '
        'refX="0" refY="3" orient="auto">\n      <path d="M0,0 L0,6 L9,3"/>\n    </marker>\n  </defs>',
    ]

    for s, p in pred:
        """
        (x1, y1) = (p_max, p_y)
        (x2, y2) = (s_min_x, s_y)
        polyline: (x1 y1, x2 y2, x3 y3, ..., xN yN)
        """
        s_min_x, s_max, s_y = arrow_points[s][:3]
        p_min_x, p_max, p_y = arrow_points[p][:3]
        p_max += 1
        mid_x = arrow_points[p][-1]  # p_max + (s_min_x - p_max) / 3
        """print(f"s_min_x: {s_min_x}, s_max: {s_max}, s_y: {s_y}")
        print(f"p_min_x: {p_min_x}, p_max: {p_max}, p_y: {p_y}")
        print(f"mid_x: {mid_x}\n")"""

        arrow = f"""  <polyline points="{p_max:.2f} {p_y:.2f}, {mid_x:.2f} {p_y:.2f}, {mid_x:.2f} {s_y:.2f}, {s_min_x - 1.:.2f} {s_y:.2f}"
                fill="none" stroke="black" stroke-width=".04" marker-end="url(#arrow)"/>"""
        if p_y != s_y:
            arrow += f'  <circle cx="{mid_x}" cy="{p_y}" r="0.1"/>'
        svg.append(arrow)
    for atoms, bonds, masks, box in render:
        molecule_svg = MoleculeContainer._graph_svg(
            atoms, bonds, masks, -font125, -box_y, width, height
        )
        molecule_svg.insert(1, box)
        svg.extend(molecule_svg)
    svg.append("</svg>")
    return "\n".join(svg)


def to_table(tree, html_path: str = "", aam: bool = False, extended=False):
    """
    Write a HTML page with the synthesis paths in SVG format and corresponding reactions in SMILES format

    :param extended:
    :param html_path: Path to save the HTML molecules_path
    :type html_path: str (optional)
    :param aam: depict atom-to-atom mapping
    :type aam: bool (optional)
    """
    if aam:
        MoleculeContainer.depict_settings(aam=True)
    else:
        MoleculeContainer.depict_settings(aam=False)

    paths = []
    if extended:
        # Gather paths
        for idx, node in tree.nodes.items():
            if node.is_solved():
                paths.append(idx)
    else:
        paths = tree.winning_nodes
    # HTML Tags
    th = '<th style="text-align: left; background-color:#978785; border: 1px solid black; border-spacing: 0">'
    td = '<td style="text-align: left; border: 1px solid black; border-spacing: 0">'
    font_red = "<font color='red' style='font-weight: bold'>"
    font_green = "<font color='light-green' style='font-weight: bold'>"
    font_head = "<font style='font-weight: bold; font-size: 18px'>"
    font_normal = "<font style='font-weight: normal; font-size: 18px'>"
    font_close = "</font>"

    template_begin = """
    <!doctype html>
    <html lang="en">
    <head>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
    rel="stylesheet"
    integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3"
    crossorigin="anonymous">
    <script
    src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"
    integrity="sha384-ka7Sk0Gln4gmtz2MlQnikT1wXgYsOg+OMhuP+IlRH9sENBO0LRn5q+8nbTov4+1p"
    crossorigin="anonymous">
    </script>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Predicted Paths Report</title>
    <meta name="description" content="A simple HTML5 Template for new projects.">
    <meta name="author" content="SitePoint">
    </head>
    <body>
    """
    template_end = """
    </body>
    </html>
    """
    # SVG Template
    box_mark = """
    <svg width="30" height="30" viewBox="0 0 1 1" xmlns="http://www.w3.org/2000/svg">
    <circle cx="0.5" cy="0.5" r="0.5" fill="rgb()" fill-opacity="0.35" />
    </svg>
    """
    # table = f"<table><thead><{th}>Retrosynthetic Routes</th></thead><tbody>"
    table = """
    <table class="table table-striped table-hover caption-top">
    <caption><h3>Retrosynthetic Routes Report</h3></caption>
    <tbody>"""

    # Gather path data
    table += f"<tr>{td}{font_normal}Target Molecule: {str(tree.nodes[1].curr_retron)}{font_close}</td></tr>"
    table += (
        f"<tr>{td}{font_normal}Tree Size: {len(tree)}{font_close} nodes</td></tr>"
    )
    table += f"<tr>{td}{font_normal}Number of visited nodes: {len(tree.visited_nodes)}{font_close}</td></tr>"
    table += f"<tr>{td}{font_normal}Found paths: {len(paths)}{font_close}</td></tr>"
    table += f"<tr>{td}{font_normal}Time: {round(tree.curr_time, 4)}{font_close} seconds</td></tr>"
    table += f"""
    <tr>{td}
                 <div>
    {box_mark.replace("rgb()", "rgb(152, 238, 255)")}
    Target Molecule
    {box_mark.replace("rgb()", "rgb(240, 171, 144)")}
    Molecule Not In Stock
    {box_mark.replace("rgb()", "rgb(155, 250, 179)")}
    Molecule In Stock
    </div>
    </td></tr>
    """

    for path in paths:
        svg = path_graph(tree, path)  # Get SVG
        full_path = tree.synthesis_path(path)  # Get Path
        # Write SMILES of all reactions in synthesis path
        step = 1
        reactions = ""
        for synth_step in full_path:
            reactions += f"<b>Step {step}:</b> {str(synth_step)}<br>"
            step += 1
        # Concatenate all content of path
        path_score = round(tree.path_score(path), 3)
        table += (
            f'<tr style="line-height: 250%">{td}{font_head}Path {path}; '
            f"Steps: {len(full_path)}; "
            f"Cumulated nodes' value: {path_score}{font_close}</td></tr>"
        )
        # f"Cumulated nodes' value: {node._probabilities[path]}{font_close}</td></tr>"
        table += f"<tr>{td}{svg}</td></tr>"
        table += f"<tr>{td}{reactions}</td></tr>"
    table += "</tbody>"
    # Save output
    if html_path:
        output = html_path
    else:
        output = tree._output
    with open(output, "w") as html_file:
        html_file.write(template_begin)
        html_file.write(table)
        html_file.write(template_end)
