import math
import textwrap

def plate_with_box_well(length, width, height, square_len, space, margin_l, margin_w, row: int, column: int):

    half_margin_l = (margin_l - square_len / 2) / 2
    half_margin_w = (margin_w - square_len / 2) / 2
    half_length = length / 2
    half_width = width / 2
    half_height = height / 2
    x_right, y_right = half_length - half_margin_l, 0.0
    x_upper, y_upper = 0.0, half_width - half_margin_w
    margin_boxes = f"""
    <geom type="box" size="{half_length} {half_margin_w} {half_height}" pos="{x_upper} {y_upper} 0" group="3" />
    <geom type="box" size="{half_length} {half_margin_w} {half_height}" pos="{x_upper} {-y_upper} 0" group="3" />
    <geom type="box" size="{half_margin_l} {half_width} {half_height}" pos="{x_right} {y_right} 0" group="3" />
    <geom type="box" size="{half_margin_l} {half_width} {half_height}" pos="{-x_right} {y_right} 0" group="3" />"""
    replicate = f"""
    <replicate count="{column - 1}" offset="{space} 0 0">
        <geom type="box" size="{(space-square_len)/2} {(half_width - half_margin_w * 2)} {half_height}" pos="-{space * (column//2 - 1)} 0 0" group="3" />
    </replicate>
    <replicate count="{row - 1}" offset="0 {space} 0">
        <geom type="box" size="{(half_length - half_margin_l * 2)} {(space-square_len)/2} {half_height}" pos="0 -{space * (row//2 - 1)} {0}" group="3" />
    </replicate>
    """
    return margin_boxes + replicate

def centrifuge_rack(length, width, height, d_small, d_big, space, row: int, column: int):
    half_length = length / 2
    half_width = width / 2
    half_height = height / 2
    half_margin_w = (width - (row - 1) * space - d_small) / 4
    half_margin_l = (length - (column - 1) * space - d_small) / 4
    x_right, y_right = half_length - half_margin_l, 0.0
    x_upper, y_upper = 0.0, half_width - half_margin_w
    margin_boxes = f"""
    <geom type="box" size="{half_length} {half_width} 0.003" pos="0 0 -0.027" group="3" />
    <geom type="box" size="{half_length} {half_margin_w} {half_height}" pos="{x_upper} {y_upper} 0" group="3" />
    <geom type="box" size="{half_length} {half_margin_w} {half_height}" pos="{x_upper} {-y_upper} 0" group="3" />
    <geom type="box" size="{half_margin_l} {half_width} {half_height}" pos="{x_right} {y_right} 0" group="3" />
    <geom type="box" size="{half_margin_l} {half_width} {half_height}" pos="{-x_right} {y_right} 0" group="3" />"""

    replicate_thin = f"""
    <replicate count="{row - 1}" offset="0 {space} 0">
        <replicate count="{column}" offset="{space} 0 0">
            <geom type="box" size="{(space-d_big)/2} {(space-d_small)/2} {half_height}" pos="-{space*(column-1)/2} -{space*(row-2)/2} 0" group="3" />
        </replicate>
    </replicate>
    """

    replicate_fat = f"""
    <replicate count="{row}" offset="0 {space} 0">
        <replicate count="{column - 1}" offset="{space} 0 0">
            <geom type="box" size="{(space-d_small)/2} {(space-d_big)/2} {half_height}" pos="-{space*(column-2)/2} -{space*(row-1)/2} 0" group="3" />
        </replicate>
    </replicate>
    """

    return margin_boxes + replicate_thin + replicate_fat

def centrifuge_plate(length, width, height, d, margin_l, margin_w, space_l, space_w, row: int, column: int):
    half_length = length / 2
    half_width = width / 2
    half_height = height / 2

    half_margin_w = (width - (row - 1) * space_w - margin_w - d / 2) / 2
    half_margin_l = (length - (column - 1) * space_l - margin_l - d / 2) / 2
    margin_boxes = f"""
    <geom type="box" size="{half_length} {(margin_w - d/2)/2} {half_height}" pos="0 -{half_width - (margin_w - d/2)/2} 0" group="3" />
    <geom type="box" size="{half_length} {half_margin_w} {half_height}" pos="0 {half_width - half_margin_w} 0" group="3" />
    <geom type="box" size="{(margin_l - d/2)/2} {half_width} {half_height}" pos="-{half_length - (margin_l - d/2)/2} 0 0" group="3" />
    <geom type="box" size="{half_margin_l} {half_width} {half_height}" pos="{half_length - half_margin_l} 0 0" group="3" />"""

    column_x = - (half_length - margin_l - space_l / 2)
    column_y = - (half_width - margin_w - space_w * (row - 1) / 2)
    column_l = space_l - d
    column_w = (row - 1) * space_w + d
    row_x = - (half_length - margin_l - space_l * (column - 1) / 2)
    row_y = - (half_width - margin_w - space_w / 2)
    row_l = (column - 1) * space_l + d
    row_w = space_w - d
    replicate = f"""
    <replicate count="{column - 1}" offset="{space_l} 0 0">
        <geom type="box" size="{column_l / 2} {column_w / 2} {half_height}" pos="{column_x} {column_y} 0" group="3" />
    </replicate>
    <replicate count="{row - 1}" offset="0 {space_w} 0">
        <geom type="box" size="{row_l / 2} {row_w / 2} {half_height}" pos="{row_x} {row_y} 0" group="3" />
    </replicate>
    """
    return margin_boxes + replicate

def tube(outer_radius, inner_radius, outer_height, inner_height, slices: int):
    bottom_half_height = (outer_height - inner_height) / 2
    bottom_radius = outer_radius

    central_angle = 2 * math.pi / slices
    half_central_angle = central_angle / 2
    side_half_thickness = (outer_radius - inner_radius) / 2 * math.cos(half_central_angle)
    side_half_width = outer_radius * math.sin(half_central_angle)
    side_half_height = inner_height / 2
    side_dist = (outer_radius + inner_radius) / 2 * math.cos(half_central_angle)
    side_pos_z = bottom_half_height * 2 + side_half_height

    result = f"""
    <geom type="cylinder" size="{bottom_radius} {bottom_half_height}" pos="0 0 {bottom_half_height}" group="3" />
    <replicate count="{slices}" euler="0 0 {-central_angle}">
        <geom type="box" size="{side_half_width} {side_half_thickness} {side_half_height}" pos="0 {side_dist} {side_pos_z}" group="3" />
    </replicate>
    """
    return result

def tube2(outer_radius, inner_radius, outer_height, inner_height, slices: int):
    bottom_half_height = (outer_height - inner_height) / 2
    bottom_radius = outer_radius

    central_angle = 2 * math.pi / slices
    half_central_angle = central_angle / 2
    side_half_thickness = (outer_radius - inner_radius) / 2 * math.cos(half_central_angle)
    side_half_width = outer_radius * math.sin(half_central_angle)
    side_half_height = inner_height / 2
    side_dist = (outer_radius + inner_radius) / 2 * math.cos(half_central_angle)
    side_pos_z = bottom_half_height * 2 + side_half_height

    outer_half_height = outer_height / 2

    result = f"""
    <geom type="cylinder" size="{bottom_radius} {bottom_half_height}" pos="0 0 {bottom_half_height}" group="3" contype="0" conaffinity="2" />
    <replicate count="{slices}" euler="0 0 {-central_angle}">
        <geom type="box" size="{side_half_width} {side_half_thickness} {side_half_height}" pos="0 {side_dist} {side_pos_z}" group="3" contype="0" conaffinity="2" />
    </replicate>
    <geom type="cylinder" size="{outer_radius} {outer_half_height}" pos="0 0 {outer_half_height}" group="4" contype="0" conaffinity="1" mass="0" />
    """
    return result

def helix(radius, pitch, low, high, gauge, slices: int):
    lead_angle = math.atan(pitch / (2 * math.pi * radius))
    central_angle = 2 * math.pi / slices
    half_central_angle = central_angle / 2
    side_half_width = radius * math.sin(half_central_angle)
    side_half_lift = pitch / slices / 2
    side_half_length = (side_half_width ** 2 + side_half_lift ** 2) ** 0.5

    replicates_high = int(high * slices)
    replicates_low = int(low * slices)
    replicates = replicates_high - replicates_low

    result = f"""
    <frame pos="0 0 {replicates_low * pitch / slices}" euler="0 0 {replicates_low * central_angle}">
        <replicate count="{replicates}" euler="0 0 {central_angle}" offset="0 0 {pitch / slices}">
            <geom type="capsule" size="{gauge} {side_half_length}" pos="{radius} 0 0" euler="{-math.pi / 2 + lead_angle} 0 0" />
        </replicate>
    </frame>
    """
    return result
    

def wrap_banner(fn, *args):
    result = textwrap.dedent(fn(*args)).strip()
    header = f"<!-- Gernerated with {fn.__name__}({', '.join(map(str, args))}) -->"
    footer = "<!-- End of generated code -->"
    return header + "\n" + result + "\n" + footer

if __name__ == "__main__":

    # generate collision mesh for tubes / bottle caps
    # params: outer_radius, inner_radius, outer_height, inner_height, slices
    # Tube:
    # |     |
    # |     |
    #  -----
    # for bottle caps, you need to flip the z-axis manually in the xml file
    print(wrap_banner(tube, 0.03, 0.028, 0.02, 0.018, 32))
    print()