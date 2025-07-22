import math

from abc import ABC, abstractmethod
from enum import Enum, IntEnum, IntFlag

from . import core
from ._util import bezier

class HitObjectType(IntFlag):
	CIRCLE          = 0b00000001
	SLIDER          = 0b00000010
	NEW_COMBO       = 0b00000100
	SPINNER         = 0b00001000
	COMBO_SKIP      = 0b01110000
	MANIA_HOLD      = 0b10000000

class HitSounds(IntFlag):
	NORMAL  = 0b0001
	WHISTLE = 0b0010
	FINISH  = 0b0100
	CLAP    = 0b1000

class SampleSet(IntEnum):
	AUTO    = 0
	NORMAL  = 1
	SOFT    = 2
	DRUM    = 3

class HitSoundExtras:
	def __init__(self, *args):
		self.sample_set = SampleSet(int(args[0]))
		self.addition_set = SampleSet(int(args[1]))
		self.customIndex = int(args[2])
		self.sampleVolume = int(args[3])
		self.filename = args[4]

class HitObject(ABC):
	def __init__(self, *args):
		self.x = int(args[0])
		self.y = int(args[1])
		self.time = int(args[2])
		self.new_combo = bool(int(args[3]) & HitObjectType.NEW_COMBO)
		self.combo_skip = int(args[3] & HitObjectType.COMBO_SKIP) >> 4
		self.hitsounds = HitSounds(int(args[4]))
		#self.extras = HitSoundExtras(*args[-1].split(":"))

	@abstractmethod
	def duration(self, beat_duration:float, multiplier:float=1.0):
		pass

	def target_position(self, time:int, beat_duration:float, multiplier:float=1.0):
		return (self.x, self.y)

class HitCircle(HitObject):
	def __init__(self, *args):
		super().__init__(*args)

	def duration(self, *args):
		return 0

class SliderType(Enum):
	LINEAR  = "L"
	BEZIER  = "B"
	PERFECT = "P"
	CATMUL  = "C"

def _compute_linear_curve(points, num_points=50):
	"""Compute linear curve points (straight lines between points)"""
	if len(points) < 2:
		return points
	
	curve_points = []
	for i in range(len(points) - 1):
		start = points[i]
		end = points[i + 1]
		
		for t in range(num_points):
			t_val = t / (num_points - 1)
			x = start[0] + t_val * (end[0] - start[0])
			y = start[1] + t_val * (end[1] - start[1])
			curve_points.append((int(x), int(y)))
	
	return curve_points


def _compute_perfect_circle_curve(points, num_points=50):
	"""Compute perfect circle curve from 3 points"""
	if len(points) < 3:
		return _compute_linear_curve(points, num_points)
	
	# For perfect circle, we need exactly 3 points
	p1, p2, p3 = points[:3]
	
	# Calculate center and radius of the circle
	# Using the perpendicular bisector method
	mid1 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
	mid2 = ((p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2)
	
	# Direction vectors
	dir1 = (p2[0] - p1[0], p2[1] - p1[1])
	dir2 = (p3[0] - p2[0], p3[1] - p2[1])
	
	# Perpendicular vectors
	perp1 = (-dir1[1], dir1[0])
	perp2 = (-dir2[1], dir2[0])
	
	# Normalize
	len1 = math.sqrt(perp1[0]**2 + perp1[1]**2)
	len2 = math.sqrt(perp2[0]**2 + perp2[1]**2)
	
	if len1 == 0 or len2 == 0:
		return _compute_linear_curve(points, num_points)
	
	perp1 = (perp1[0] / len1, perp1[1] / len1)
	perp2 = (perp2[0] / len2, perp2[1] / len2)
	
	# Find intersection of perpendicular bisectors
	# This is the center of the circle
	try:
		# Solve for intersection point
		# mid1 + t1 * perp1 = mid2 + t2 * perp2
		det = perp1[0] * perp2[1] - perp1[1] * perp2[0]
		if abs(det) < 1e-10:
			return _compute_linear_curve(points, num_points)
		
		t1 = ((mid2[0] - mid1[0]) * perp2[1] - (mid2[1] - mid1[1]) * perp2[0]) / det
		center = (mid1[0] + t1 * perp1[0], mid1[1] + t1 * perp1[1])
		
		# Calculate radius
		radius = math.sqrt((p1[0] - center[0])**2 + (p1[1] - center[1])**2)
		
		# Calculate angles for all three points
		angle1 = math.atan2(p1[1] - center[1], p1[0] - center[0])
		angle2 = math.atan2(p2[1] - center[1], p2[0] - center[0])
		angle3 = math.atan2(p3[1] - center[1], p3[0] - center[0])
		
		# Determine the correct arc direction from angle1 to angle3 that passes through angle2
		# We need to find which direction (clockwise or counterclockwise) from angle1 to angle3
		# naturally passes through angle2
		
		# Normalize angles to [0, 2π] for easier calculations
		def normalize_angle(angle):
			while angle < 0:
				angle += 2 * math.pi
			while angle >= 2 * math.pi:
				angle -= 2 * math.pi
			return angle
		
		angle1_norm = normalize_angle(angle1)
		angle2_norm = normalize_angle(angle2)
		angle3_norm = normalize_angle(angle3)
		
		# Check if angle2 lies on the arc from angle1 to angle3 (counterclockwise)
		def angle_between_ccw(start, middle, end):
			# Check if middle angle is between start and end when going counterclockwise
			if start <= end:
				return start <= middle <= end
			else:  # Arc crosses 0
				return middle >= start or middle <= end
		
		# Check if angle2 lies on the arc from angle1 to angle3 (clockwise)
		def angle_between_cw(start, middle, end):
			# Check if middle angle is between start and end when going clockwise
			if start >= end:
				return start >= middle >= end
			else:  # Arc crosses 0 going backwards
				return middle <= start or middle >= end
		
		# Determine the correct direction
		if angle_between_ccw(angle1_norm, angle2_norm, angle3_norm):
			# Counterclockwise direction
			final_angle = angle3
			if angle3 < angle1:
				final_angle = angle3 + 2 * math.pi
		elif angle_between_cw(angle1_norm, angle2_norm, angle3_norm):
			# Clockwise direction  
			final_angle = angle3
			if angle3 > angle1:
				final_angle = angle3 - 2 * math.pi
		else:
			# Fallback: use shorter arc direction
			ccw_diff = angle3_norm - angle1_norm
			if ccw_diff < 0:
				ccw_diff += 2 * math.pi
			
			cw_diff = angle1_norm - angle3_norm
			if cw_diff < 0:
				cw_diff += 2 * math.pi
			
			if ccw_diff <= cw_diff:
				# Counterclockwise is shorter
				final_angle = angle3
				if angle3 < angle1:
					final_angle = angle3 + 2 * math.pi
			else:
				# Clockwise is shorter
				final_angle = angle3
				if angle3 > angle1:
					final_angle = angle3 - 2 * math.pi
		
		# Generate points along the single continuous arc
		curve_points = []
		total_angle_diff = final_angle - angle1
		
		for i in range(num_points):
			t_val = i / (num_points - 1) if num_points > 1 else 0
			angle = angle1 + t_val * total_angle_diff
			x = center[0] + radius * math.cos(angle)
			y = center[1] + radius * math.sin(angle)
			curve_points.append((int(x), int(y)))
		
		return curve_points
	except:
		return _compute_linear_curve(points, num_points)


def _compute_catmull_rom_curve(points, num_points=50):
	"""Compute Catmull-Rom curve points"""
	if len(points) < 3:
		return _compute_linear_curve(points, num_points)
	
	curve_points = []
	
	# Add virtual start and end points for Catmull-Rom
	extended_points = [points[0]] + list(points) + [points[-1]]
	
	for i in range(1, len(extended_points) - 2):
		p0 = extended_points[i - 1]
		p1 = extended_points[i]
		p2 = extended_points[i + 1]
		p3 = extended_points[i + 2]
		
		for t in range(num_points):
			t_val = t / (num_points - 1)
			
			# Catmull-Rom matrix coefficients
			t2 = t_val * t_val
			t3 = t2 * t_val
			
			# Catmull-Rom blending functions
			b0 = -0.5 * t3 + t2 - 0.5 * t_val
			b1 = 1.5 * t3 - 2.5 * t2 + 1
			b2 = -1.5 * t3 + 2 * t2 + 0.5 * t_val
			b3 = 0.5 * t3 - 0.5 * t2
			
			x = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
			y = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
			
			curve_points.append((int(x), int(y)))
	
	return curve_points


class Slider(HitCircle):
	def __init__(self, *args):
		super().__init__(*args)

		slider_info = args[5].split("|")
		self.slider_type = SliderType(slider_info[0])
		self.curve_points = list(map(lambda p: tuple(map(int, p)), [p.split(':') for p in slider_info[1:]]))

		self.repeat = int(args[6])
		self.pixel_length = int(args[7])
		#self.edge_hitsounds = [HitSounds(int(h)) for h in args[8].split('|')]

		#additions = [e.split(":") for e in args[9].split('|')]
		#self.edge_additions = [(SampleSet(int(s)), SampleSet(int(a))) for s, a in additions]

	def duration(self, beat_duration:float, multiplier:float=1.0):
		return beat_duration * self.pixel_length / (100 * multiplier) * self.repeat

	def real_curve_points(self):
		points = []
		for i in range(1, self.repeat + 1):
			l = ([(self.x, self.y)] + self.curve_points)
			if i % 2 == 0:
				l = list(reversed(l))
			points += l
		return points

	def _get_base_curve_points(self):
		"""Get the base curve points for the slider (without repeats)"""
		all_points = [(self.x, self.y)] + self.curve_points
		
		# Compute base curve based on slider type
		if self.slider_type == SliderType.BEZIER:
			base_curve = bezier.compute(all_points)
		elif self.slider_type == SliderType.LINEAR:
			base_curve = _compute_linear_curve(all_points)
		elif self.slider_type == SliderType.PERFECT:
			base_curve = _compute_perfect_circle_curve(all_points)
		elif self.slider_type == SliderType.CATMUL:
			base_curve = _compute_catmull_rom_curve(all_points)
		else:
			# Fallback to linear
			base_curve = _compute_linear_curve(all_points)
		
		# Calculate the actual length of the curve
		curve_length = 0
		segment_lengths = []
		for i in range(1, len(base_curve)):
			dx = base_curve[i][0] - base_curve[i-1][0]
			dy = base_curve[i][1] - base_curve[i-1][1]
			segment_length = math.sqrt(dx*dx + dy*dy)
			curve_length += segment_length
			segment_lengths.append(segment_length)
		
		# Handle length adjustment
		if len(base_curve) > 1:
			if curve_length < self.pixel_length:
				# Extend the curve with a straight line
				remaining_length = self.pixel_length - curve_length
				
				# Get the direction of the last segment
				last_point = base_curve[-1]
				second_last_point = base_curve[-2]
				
				dx = last_point[0] - second_last_point[0]
				dy = last_point[1] - second_last_point[1]
				last_segment_length = math.sqrt(dx*dx + dy*dy)
				
				if last_segment_length > 0:
					# Normalize the direction vector
					dx = dx / last_segment_length
					dy = dy / last_segment_length
					
					# Calculate the end point of the extension
					extension_x = int(last_point[0] + dx * remaining_length)
					extension_y = int(last_point[1] + dy * remaining_length)
					
					# Add the extension point
					base_curve.append((extension_x, extension_y))
			
			elif curve_length > self.pixel_length:
				# Shorten the curve to match the target length
				target_length = self.pixel_length
				current_length = 0
				shortened_curve = [base_curve[0]]  # Always include the start point
				
				for i in range(1, len(base_curve)):
					dx = base_curve[i][0] - base_curve[i-1][0]
					dy = base_curve[i][1] - base_curve[i-1][1]
					segment_length = math.sqrt(dx*dx + dy*dy)
					
					if current_length + segment_length <= target_length:
						# Include the full segment
						shortened_curve.append(base_curve[i])
						current_length += segment_length
					else:
						# Partial segment - interpolate to the target length
						remaining_length = target_length - current_length
						if segment_length > 0:
							ratio = remaining_length / segment_length
							interpolated_x = int(base_curve[i-1][0] + dx * ratio)
							interpolated_y = int(base_curve[i-1][1] + dy * ratio)
							shortened_curve.append((interpolated_x, interpolated_y))
						break
				
				base_curve = shortened_curve
		
		return base_curve

	def _get_curve_points(self):
		"""Get the complete curve points for the slider including repeats"""
		base_curve = self._get_base_curve_points()
		
		# Handle repeats and reversals
		complete_curve = []
		for i in range(1, self.repeat + 1):
			if i % 2 == 1:
				# Forward direction
				complete_curve.extend(base_curve)
			else:
				# Reverse direction
				complete_curve.extend(list(reversed(base_curve)))
		
		return complete_curve

	def current_curve_point(self, time:int, beat_duration:float, multiplier:float=1.0):
		elapsed = time - self.time
		if elapsed <= 0:
			return (self.x, self.y)

		duration = self.duration(beat_duration, multiplier)
		if elapsed >= duration:
			# Return the end position
			curve_points = self._get_curve_points()
			return curve_points[-1] if curve_points else (self.x, self.y)

		# Calculate progress along the slider (0.0 to 1.0)
		progress = elapsed / duration
		
		# Get the complete curve points including repeats
		curve_points = self._get_curve_points()
		
		if not curve_points:
			return (self.x, self.y)
		
		# Calculate the position along the curve using arc-length parameterization
		total_points = len(curve_points)
		if total_points <= 1:
			return curve_points[0]
		
		# Calculate cumulative arc lengths for each point
		cumulative_lengths = [0.0]
		total_length = 0.0
		
		for i in range(1, len(curve_points)):
			dx = curve_points[i][0] - curve_points[i-1][0]
			dy = curve_points[i][1] - curve_points[i-1][1]
			segment_length = math.sqrt(dx*dx + dy*dy)
			total_length += segment_length
			cumulative_lengths.append(total_length)
		
		# Find the target arc length based on progress
		target_length = progress * total_length
		
		# Find the segment containing the target length
		for i in range(len(cumulative_lengths) - 1):
			if cumulative_lengths[i] <= target_length <= cumulative_lengths[i + 1]:
				# Interpolate within this segment
				segment_start_length = cumulative_lengths[i]
				segment_end_length = cumulative_lengths[i + 1]
				segment_length = segment_end_length - segment_start_length
				
				if segment_length > 0:
					# Calculate interpolation ratio within the segment
					segment_progress = (target_length - segment_start_length) / segment_length
					
					# Linear interpolation between the two points
					start_point = curve_points[i]
					end_point = curve_points[i + 1]
					
					x = start_point[0] + segment_progress * (end_point[0] - start_point[0])
					y = start_point[1] + segment_progress * (end_point[1] - start_point[1])
					
					return (int(x), int(y))
				else:
					# Zero-length segment, return the point
					return curve_points[i]
		
		# Fallback: return the last point
		return curve_points[-1]

	def target_position(self, time: int, beat_duration: float, multiplier: float=1.0):
		return self.current_curve_point(time, beat_duration, multiplier)

class Spinner(HitObject):
	def __init__(self, *args):
		super().__init__(*args)
		self.end_time = int(args[5])
		
	def duration(self, *args):
		return self.end_time - self.time

def create(obj):
	if obj[3] & HitObjectType.CIRCLE:
		return HitCircle(*obj)
	elif obj[3] & HitObjectType.SLIDER:
		return Slider(*obj)
	elif obj[3] & HitObjectType.SPINNER:
		return Spinner(*obj)