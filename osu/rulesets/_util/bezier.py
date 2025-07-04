# https://www.pygame.org/wiki/BezierCurve

def compute(vertices, numPoints=None):
    """Compute Bézier curve points from control points"""
    if numPoints is None:
        numPoints = 50
    
    if len(vertices) < 2:
        return vertices
    
    # For Bézier curves with arbitrary number of points, we need to handle them differently
    # osu! uses a special format where multiple Bézier curves can be joined
    curve_points = []
    
    # Find segments (osu! Bézier curves are separated by repeated points)
    segments = []
    current_segment = [vertices[0]]
    
    for i in range(1, len(vertices)):
        if vertices[i] == vertices[i-1] and len(current_segment) > 1:
            # End of segment, start new one
            segments.append(current_segment)
            current_segment = [vertices[i]]
        else:
            current_segment.append(vertices[i])
    
    # Add the last segment
    if current_segment:
        segments.append(current_segment)
    
    # Process each segment
    for segment in segments:
        if len(segment) == 1:
            curve_points.append(segment[0])
        elif len(segment) == 2:
            # Linear interpolation for 2 points
            for t in range(numPoints):
                t_val = t / (numPoints - 1)
                x = segment[0][0] + t_val * (segment[1][0] - segment[0][0])
                y = segment[0][1] + t_val * (segment[1][1] - segment[0][1])
                curve_points.append((int(x), int(y)))
        elif len(segment) == 3:
            # Quadratic Bézier curve
            for t in range(numPoints):
                t_val = t / (numPoints - 1)
                # Quadratic Bézier formula: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
                one_minus_t = 1 - t_val
                x = one_minus_t**2 * segment[0][0] + 2 * one_minus_t * t_val * segment[1][0] + t_val**2 * segment[2][0]
                y = one_minus_t**2 * segment[0][1] + 2 * one_minus_t * t_val * segment[1][1] + t_val**2 * segment[2][1]
                curve_points.append((int(x), int(y)))
        elif len(segment) == 4:
            # Cubic Bézier curve - use the original algorithm
            segment_curve = _compute_cubic_bezier(segment, numPoints)
            if segment_curve:
                curve_points.extend(segment_curve)
            else:
                # Fallback to linear
                for t in range(numPoints):
                    t_val = t / (numPoints - 1)
                    x = segment[0][0] + t_val * (segment[3][0] - segment[0][0])
                    y = segment[0][1] + t_val * (segment[3][1] - segment[0][1])
                    curve_points.append((int(x), int(y)))
        else:
            # Higher degree Bézier curve - use de Casteljau's algorithm
            for t in range(numPoints):
                t_val = t / (numPoints - 1)
                point = _de_casteljau(segment, t_val)
                curve_points.append((int(point[0]), int(point[1])))
    
    return curve_points


def _compute_cubic_bezier(vertices, numPoints=30):
    """Original cubic Bézier curve computation for 4 control points"""
    if numPoints < 2 or len(vertices) != 4:
        return None

    result = []

    b0x = vertices[0][0]
    b0y = vertices[0][1]
    b1x = vertices[1][0]
    b1y = vertices[1][1]
    b2x = vertices[2][0]
    b2y = vertices[2][1]
    b3x = vertices[3][0]
    b3y = vertices[3][1]

    # Compute polynomial coefficients from Bezier points
    ax = -b0x + 3 * b1x + -3 * b2x + b3x
    ay = -b0y + 3 * b1y + -3 * b2y + b3y

    bx = 3 * b0x + -6 * b1x + 3 * b2x
    by = 3 * b0y + -6 * b1y + 3 * b2y

    cx = -3 * b0x + 3 * b1x
    cy = -3 * b0y + 3 * b1y

    dx = b0x
    dy = b0y

    # Set up the number of steps and step size
    numSteps = numPoints - 1 # arbitrary choice
    h = 1.0 / numSteps # compute our step size

    # Compute forward differences from Bezier points and "h"
    pointX = dx
    pointY = dy

    firstFDX = ax * (h * h * h) + bx * (h * h) + cx * h
    firstFDY = ay * (h * h * h) + by * (h * h) + cy * h

    secondFDX = 6 * ax * (h * h * h) + 2 * bx * (h * h)
    secondFDY = 6 * ay * (h * h * h) + 2 * by * (h * h)

    thirdFDX = 6 * ax * (h * h * h)
    thirdFDY = 6 * ay * (h * h * h)

    # Compute points at each step
    result.append((int(pointX), int(pointY)))

    for _ in range(numSteps):
        pointX += firstFDX
        pointY += firstFDY

        firstFDX += secondFDX
        firstFDY += secondFDY

        secondFDX += thirdFDX
        secondFDY += thirdFDY

        result.append((int(pointX), int(pointY)))

    return result


def _de_casteljau(points, t):
    """De Casteljau's algorithm for computing Bézier curves of arbitrary degree"""
    if len(points) == 1:
        return points[0]
    
    new_points = []
    for i in range(len(points) - 1):
        x = (1 - t) * points[i][0] + t * points[i + 1][0]
        y = (1 - t) * points[i][1] + t * points[i + 1][1]
        new_points.append((x, y))
    
    return _de_casteljau(new_points, t)