import numpy as np
import random

from fractions import Fraction


class GeometryUtils:
    @staticmethod
    def generate_point(limit=10):
        return np.random.randint(-limit, limit, 3)

    @staticmethod
    def generate_direction(limit=10):
        direction = np.zeros(3)
        while np.all(direction == 0):
            direction = np.random.randint(-limit, limit, 3)
        return direction

    @staticmethod
    def generate_plane_through_line(point, line_direction, limit=10):
        in_plane_vector = GeometryUtils.generate_direction(limit)
        normal_vector = np.cross(line_direction, in_plane_vector)
        if np.all(normal_vector == 0):
            normal_vector = np.cross(line_direction, np.array([1, 0, 0]))
        return point, normal_vector

    @staticmethod
    def format_equation(coefficients, constant):
        terms = []
        for i, coef in enumerate(coefficients):
            if coef != 0:
                if terms:
                    terms.append(f"{coef:+g}{'xyz'[i]}")
                else:
                    terms.append(f"{coef:g}{'xyz'[i]}")
        equation = " ".join(terms).replace("+-", "-")
        equation = equation.replace("-  -", "-").replace("+  +", "+")
        return (
            f"{equation} {constant:+g}".replace("+", "+ ").replace("-", "- ").strip()
            + " = 0"
        )

    @staticmethod
    def format_parametric_line(point, direction):
        terms = []
        for i, (p, d) in enumerate(zip(point, direction)):
            if p != 0 or d != 0:
                if d != 0:
                    if p != 0:
                        terms.append(
                            f"{'xyz'[i]} = {p} {'+' if d > 0 else '-'} {abs(d)}t"
                        )
                    else:
                        terms.append(
                            f"{'xyz'[i]} = {0} {'+' if d > 0 else '-'} {abs(d)}t"
                        )
                else:
                    terms.append(f"{'xyz'[i]} = {p}")
        return ", ".join(terms).replace("+-", "-").replace("- -", "- ")

    @staticmethod
    def format_canonical_equation(point, direction):
        terms = []
        for i, (p, d) in enumerate(zip(point, direction)):
            if d == 0:
                terms.append(f"\\frac{{{'xyz'[i]} - ({p})}}{{0}}")
            else:
                point_str = f"{p}" if p >= 0 else f"({p})"
                terms.append(f"\\frac{{{'xyz'[i]} - {point_str}}}{{{d}}}")
        return " = ".join(terms).replace("+-", "-")

    @staticmethod
    def check_parallel(line_direction, plane_normal):
        return np.isclose(np.dot(line_direction, plane_normal), 0)

    @staticmethod
    def check_point_on_plane(plane_point, line_point, plane_normal):
        D = -np.dot(plane_normal, plane_point)
        return np.isclose(np.dot(plane_normal, line_point) + D, 0)


def format_parametric_plane(point, direction1, direction2):
    terms = []
    for i, p in enumerate(point):
        term = f"{'xyz'[i]} = {p}"
        if direction1[i] != 0:
            term += f" {'+' if direction1[i] > 0 else '-'} {abs(direction1[i])}u"
        if direction2[i] != 0:
            term += f" {'+' if direction2[i] > 0 else '-'} {abs(direction2[i])}v"
        terms.append(term)
    return (
        ", ".join(terms)
        .replace("+-", "-")
        .replace("- -", "- ")
        .replace(" + 0u", "")
        .replace(" + 0v", "")
    )


def parametric_equations(point, normal):
    point_on_plane = find_point_on_plane(point, normal)
    direction1, direction2 = find_directions(normal)
    return format_parametric_plane(point_on_plane, direction1, direction2)


def find_point_on_plane(point, normal, limit=10):
    while True:
        rand_point = generate_point(limit)
        if (
            normal[0] * rand_point[0]
            + normal[1] * rand_point[1]
            + normal[2] * rand_point[2]
            - np.dot(normal, point)
            == 0
        ):
            return point


def find_directions(normal, limit=10):
    normal = np.array(normal)
    while True:
        direction1 = generate_direction(limit)
        if np.dot(normal, direction1) == 0:
            break
    while True:
        direction2 = generate_direction(limit)
        if np.dot(normal, direction2) == 0 and not np.all(direction1 == direction2):
            return direction1, direction2


def generate_point(limit=50):
    return np.array([generate_rational(limit) for _ in range(3)], dtype=object)


def generate_direction(limit=50):
    while True:
        direction = np.array(
            [generate_rational(limit) for _ in range(3)], dtype=object
        )
        if not np.all(direction == 0):
            return direction


def generate_rational(limit=50):
    numerator = random.randint(-limit, limit)
    denominator = random.randint(1, 10)
    return Fraction(numerator, denominator)


class Line:
    def __init__(self, point, direction):
        self.point = point
        self.direction = direction

    def equations(self):
        if self.direction[0] != 0:
            normal1 = np.array([self.direction[1], -self.direction[0], 0])
        else:
            normal1 = np.array([0, self.direction[2], -self.direction[1]])
        normal2 = np.cross(self.direction, normal1)
        D1 = -np.dot(normal1, self.point)
        D2 = -np.dot(normal2, self.point)

        general_eq = f"{GeometryUtils.format_equation(normal1, D1)} \\text{{ и }} {GeometryUtils.format_equation(normal2, D2)}"
        parametric_eq = GeometryUtils.format_parametric_line(self.point, self.direction)
        canonical_eq = GeometryUtils.format_canonical_equation(
            self.point, self.direction
        )

        return general_eq, parametric_eq, canonical_eq


class Plane:
    def __init__(self, point, normal):
        self.point = point
        self.normal = normal

    def equations(self):
        D = -np.dot(self.normal, self.point)
        general_eq = GeometryUtils.format_equation(self.normal, D)
        parametric_eq1 = parametric_equations(self.point, self.normal)
        return general_eq, parametric_eq1


class GeometryGenerator:
    @staticmethod
    def generate_parallel_line_and_plane():
        while True:
            plane_point = GeometryUtils.generate_point()
            line_point = GeometryUtils.generate_point()
            line_direction = GeometryUtils.generate_direction()
            plane_normal = GeometryUtils.generate_direction()
            if GeometryUtils.check_parallel(
                line_direction, plane_normal
            ) and not GeometryUtils.check_point_on_plane(
                plane_point, line_point, plane_normal
            ):
                return Plane(plane_point, plane_normal), Line(
                    line_point, line_direction
                )

    @staticmethod
    def generate_line_on_plane():
        while True:
            plane_point = GeometryUtils.generate_point()
            line_point = GeometryUtils.generate_point()
            line_direction = GeometryUtils.generate_direction()
            plane_normal = GeometryUtils.generate_direction()
            if GeometryUtils.check_point_on_plane(
                plane_point, line_point, plane_normal
            ) and GeometryUtils.check_parallel(line_direction, plane_normal):
                return Plane(plane_point, plane_normal), Line(
                    line_point, line_direction
                )

    @staticmethod
    def generate_and_format(case_generator, num_variants=1):
        results = []

        for i in range(num_variants):
            plane, line = case_generator()
            general_line_eq, parametric_line_eq, canonical_line_eq = line.equations()
            general_plane_eq, parametric_plane_eq = plane.equations()
            results.append(f"**Вариант {i+1}:**")
            results.append(f"Точка на плоскости: {plane.point}")
            results.append(f"Точка на прямой: {line.point}")
            results.append(f"Направляющий вектор прямой: {line.direction}")
            results.append(f"Нормальный вектор плоскости: {plane.normal}")
            results.append(f"Общее уравнение прямой: ${general_line_eq}$")
            results.append(f"Параметрическое уравнение прямой: ${parametric_line_eq}$")
            results.append(f"Каноническое уравнение прямой: ${canonical_line_eq}$")
            results.append(f"Общее уравнение плоскости: ${general_plane_eq}$")
            results.append(
                f"Параметрическое уравнение плоскости: ${parametric_plane_eq}$"
            )

        return "\n\n".join(results)

    @staticmethod
    def generate_variants(num_variants=1, txt=False):
        results = []
        results.append("### Пересечение прямой и плоскости")
        results.append(
            GeometryGenerator.generate_and_format(
                lambda: (
                    Plane(
                        GeometryUtils.generate_point(),
                        GeometryUtils.generate_plane_through_line(
                            GeometryUtils.generate_point(),
                            GeometryUtils.generate_direction(),
                        )[1],
                    ),
                    Line(
                        GeometryUtils.generate_point(),
                        GeometryUtils.generate_direction(),
                    ),
                ),
                num_variants,
            )
        )
        results.append("### Параллельность прямой и плоскости")
        results.append(
            GeometryGenerator.generate_and_format(
                GeometryGenerator.generate_parallel_line_and_plane, num_variants
            )
        )
        results.append("### Принадлежность прямой плоскости")
        results.append(
            GeometryGenerator.generate_and_format(
                GeometryGenerator.generate_line_on_plane, num_variants
            )
        )
        result_str = "\n\n".join(results)
        if txt:
            with open("line_and_plane.txt", "w", encoding="utf-8") as file:
                file.write(result_str)

        return "\n\n".join(results)

class Planes:
    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def __call__(self):
        return [self.A, self.B, self.C, self.D]

    def equation(self):
        terms = []
        for i, coef in enumerate([self.A, self.B, self.C]):
            if coef != 0:
                if terms:
                    terms.append(f"{coef:+g}{'xyz'[i]}")
                else:
                    terms.append(f"{coef:g}{'xyz'[i]}")
        equation = " ".join(terms).replace("+-", "-")
        equation = equation.replace("-  -", "-").replace("+  +", "+")
        return f"{equation} {self.D:+g}".replace("+", "+ ").replace("-", "- ").strip() + " = 0"


    def parametric_equations(self):
        point_on_plane = self.find_point_on_plane()
        direction1, direction2 = self.find_directions()
        return self._format_parametric_equation(point_on_plane, direction1, direction2)

    def find_point_on_plane(self, limit=10):
        while True:
            point = self.generate_point(limit)
            if self.A * point[0] + self.B * point[1] + self.C * point[2] + self.D == 0:
                return point

    def find_directions(self, limit=10):
        normal = np.array([self.A, self.B, self.C])
        while True:
            direction1 = self.generate_direction(limit)
            if np.dot(normal, direction1) == 0:
                break
        while True:
            direction2 = self.generate_direction(limit)
            if np.dot(normal, direction2) == 0 and not np.all(direction1 == direction2):
                return direction1, direction2

    def generate_point(self, limit=50):
        return np.array([self.generate_rational(limit) for _ in range(3)], dtype=object)

    def generate_direction(self, limit=50):
        while True:
            direction = np.array([self.generate_rational(limit) for _ in range(3)], dtype=object)
            if not np.all(direction == 0):
                return direction

    def generate_rational(self, limit=50):
        numerator = random.randint(-limit, limit)
        denominator = random.randint(1, 10)
        return Fraction(numerator, denominator)
    
    def _format_parametric_equation(self, point, direction1, direction2):
        terms = []
        for i, p in enumerate(point):
            term = f"{'xyz'[i]} = {p}"
            if direction1[i] != 0:
                term += f" {'+' if direction1[i] > 0 else '-'} {abs(direction1[i])}u"
            if direction2[i] != 0:
                term += f" {'+' if direction2[i] > 0 else '-'} {abs(direction2[i])}v"
            terms.append(term)
        return ", ".join(terms).replace("+-", "-").replace("- -", "- ").replace(" + 0u", "").replace(" + 0v", "")

def check_proportionality(row1, row2):
    ratios = []
    for a, b in zip(row1, row2):
        if b != 0:
            ratios.append(a / b)
        elif a != 0:
            return False
    return all(np.isclose(ratio, ratios[0]) for ratio in ratios)

def check_all_pairs_proportionality(matrix):
    n = matrix.shape[0]
    proportional_pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            if check_proportionality(matrix[i], matrix[j]):
                proportional_pairs.append((i, j))

    return proportional_pairs

def identify_case(rank_A, rank_A_prime, matrix_A, matrix_A1):
    cases = {
        (3, 3): 1,
        (2, 2): 2 if not check_all_pairs_proportionality(matrix_A) else 3,
        (1, 1): 4,
        (2, 3): 5 if not check_all_pairs_proportionality(matrix_A) else 6,
        (1, 2): 7 if check_all_pairs_proportionality(matrix_A1) else 8
    }
    return cases.get((rank_A, rank_A_prime))

def generate_integer_matrix_with_exact_rank(rows, cols, rank):
    while True:
        U = np.random.randint(-3, 4, size=(rows, rank))
        V = np.random.randint(-3, 4, size=(cols, rank))
        S_values = np.random.randint(1, 3, size=rank)
        A = np.dot(U * S_values, V.T)
        if not np.any(np.all(A == 0, axis=1)):
            return A

def matrix_to_planes(matrix):
    if matrix.shape[1] != 4:
        raise ValueError("Each row must have exactly 4 elements to initialize a Plane.")
    return [Planes(*row) for row in matrix]

def generate_matrices():
    ranks = [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]
    rank1, rank2 = random.choice(ranks)
    matrix_A = generate_integer_matrix_with_exact_rank(3, 3, rank1)
    matrix_A1 = np.zeros((3, 4), dtype=int)
    matrix_A1[:, :3] = matrix_A

    while True:
        matrix_A1[:, 3] = np.random.randint(-3, 4, size=3)
        if np.linalg.matrix_rank(matrix_A1) == rank2:
            break

    return matrix_A, matrix_A1, rank1, rank2
    
@staticmethod
def generate_planes(set_number, txt=False):
    if set_number < 8:
        raise ValueError("set_number cannot be less than 8")

    cases = {i: [] for i in range(1, 9)}
    results = []
    unique_num = set_number // 8
    unique_cases_filled = False

    while len(results) < set_number:
        matrix_A, matrix_A1, rank_A, rank_A1 = generate_matrices()
        case_description = identify_case(rank_A, rank_A1, matrix_A, matrix_A1)
        if case_description is None:
            continue

        plane1, plane2, plane3 = matrix_to_planes(matrix_A1)
        param1, param2, param3 = plane1.parametric_equations(), plane2.parametric_equations(), plane3.parametric_equations()

        if not unique_cases_filled:
            if len(cases[case_description]) < unique_num:
                cases[case_description].append((plane1, plane2, plane3, rank_A, rank_A1, case_description))
                results.append(
                    (
                        case_description,
                        f"### Set: {len(results) + 1}, Case {case_description}: Rank A = {rank_A}, Rank A' = {rank_A1}\n"
                        f"Plane 1 general: {plane1.equation()}\n\n"
                        f"Plane 1 parametric: {param1}\n\n"
                        f"Plane 2 general: {plane2.equation()}\n\n"
                        f"Plane 2 parametric: {param2}\n\n"
                        f"Plane 3 general: {plane3.equation()}\n\n"
                        f"Plane 3 parametric: {param3}\n\n\n"
                    )
                )

            unique_cases_filled = all(len(cases[i]) >= unique_num for i in range(1, 9))
        else:
            if len(results) < set_number:
                cases[case_description].append((plane1, plane2, plane3, rank_A, rank_A1, case_description))
                results.append(
                    (
                        case_description,
                        f"### Set: {len(results) + 1}, Case {case_description}: Rank A = {rank_A}, Rank A' = {rank_A1}\n"
                        f"Plane 1 general: {plane1.equation()}\n\n"
                        f"Plane 1 parametric: {param1}\n\n"
                        f"Plane 2 general: {plane2.equation()}\n\n"
                        f"Plane 2 parametric: {param2}\n\n"
                        f"Plane 3 general: {plane3.equation()}\n\n"
                        f"Plane 3 parametric: {param3}\n\n\n"
                    )
                )

    results.sort(key=lambda x: x[0])
    result_str = "\n".join([result[1] for result in results])

    if txt:
        with open("planes.txt", "w", encoding="utf-8") as file:
            file.write(result_str)
            
def generate_random_point():
    return np.random.randint(-10, 11, size=3)


def generate_random_direction():
    while True:
        direction = np.random.randint(-10, 11, size=3)
        if np.linalg.norm(direction) != 0:
            return direction


def format_canonical_equation(point, direction):
    terms = []
    for i, (p, d) in enumerate(zip(point, direction)):
        if d == 0:
            terms.append(f"\\frac{{{'xyz'[i]} - ({p})}}{{0}}")
        else:
            point_str = f"{p}" if p >= 0 else f"({p})"
            terms.append(f"\\frac{{{'xyz'[i]} - {point_str}}}{{{d}}}")
    return " = ".join(terms).replace("+-", "-")

    
class Line:
        def __init__(self, point, direction):
            self.point = np.array(point)
            self.direction = np.array(direction)
    
        def __call__(self):
            return self.point, self.direction
    
        def equations(self):
            if self.direction[0] != 0:
                normal1 = np.array([self.direction[1], -self.direction[0], 0])
            else:
                normal1 = np.array([0, self.direction[2], -self.direction[1]])
            normal2 = np.cross(self.direction, normal1)
            D1 = -np.dot(normal1, self.point)
            D2 = -np.dot(normal2, self.point)
    
            general_eq = f"{GeometryUtils.format_equation(normal1, D1)} \\text{{ and }} {GeometryUtils.format_equation(normal2, D2)}"
            parametric_eq = GeometryUtils.format_parametric_line(
                self.point, self.direction
            )
            canonical_eq = GeometryUtils.format_canonical_equation(
                self.point, self.direction
            )
    
            return general_eq, parametric_eq, canonical_eq
    
        def equation(self):
            return f"Point: {self.point}, Direction: {self.direction}"
    
        def contains_point(self, point):
            direction_nonzero = self.direction != 0
            t_values = (
                point[direction_nonzero] - self.point[direction_nonzero]
            ) / self.direction[direction_nonzero]
            return np.all(np.isclose(t_values, t_values[0]))
    
    
    def check_complanar(v1, v2, v3):
        return np.isclose(np.dot(np.cross(v1, v2), v3), 0)
    
    
    def check_collinear(v1, v2):
        cross_product = np.cross(v1, v2)
        return np.allclose(cross_product, 0)
    
    
    def lines_relationship(line1, line2, g):
        v1 = line1.direction
        v2 = line2.direction
        p1 = line1.point
        p2 = line2.point
    
        v3 = p2 - p1
    
        if not check_complanar(g, v1, v2):
            return "skew"
        if not check_collinear(v1, v2):
            return "intersecting"
    
        if line1.contains_point(p2):
            return "coincident"
    
        return "parallel"
    
    
    def generate_lines(num_variants=30, txt=False):
        if num_variants < 4:
            raise ValueError("num_variants cannot be less than 4")
    
        cases = {"skew": [], "intersecting": [], "coincident": [], "parallel": []}
        results = []
        unique_num = num_variants // 4
        unique_cases_filled = False
    
        while len(results) < num_variants:
            # Генерация случайных точек и направляющих векторов
            A = generate_random_point()
            B = generate_random_point()
            p1 = generate_random_direction()
            p2 = generate_random_direction()
    
            # Вычисление вектора g
            g = A - B
    
            line1 = Line(A, p1)
            line2 = Line(B, p2)
    
            relationship = lines_relationship(line1, line2, g)
    
            if not unique_cases_filled:
                if len(cases[relationship]) < unique_num:
                    cases[relationship].append((line1, line2))
    
                    general_eq1, parametric_eq1, canonical_eq1 = line1.equations()
                    general_eq2, parametric_eq2, canonical_eq2 = line2.equations()
    
                    results.append(
                        (
                            relationship,
                            f"### Set: {len(results) + 1}, Lines are {relationship}:\n"
                            f"Line 1 general eq: ${general_eq1}$\n\n"
                            f"Line 1 parametric eq: ${parametric_eq1}$\n\n"
                            f"Line 1 canonical eq: ${canonical_eq1}$\n\n"
                            f"Line 2 general eq: ${general_eq2}$\n\n"
                            f"Line 2 parametric eq: ${parametric_eq2}$\n\n"
                            f"Line 2 canonical eq: ${canonical_eq2}$\n\n\n",
                        )
                    )
    
                unique_cases_filled = all(len(cases[k]) >= unique_num for k in cases.keys())
            else:
                if len(results) < num_variants:
                    cases[relationship].append((line1, line2))
    
                    general_eq1, parametric_eq1, canonical_eq1 = line1.equations()
                    general_eq2, parametric_eq2, canonical_eq2 = line2.equations()
    
                    results.append(
                        (
                            relationship,
                            f"### Set: {len(results) + 1}, Lines are {relationship}:\n"
                            f"Line 1 general eq: ${general_eq1}$\n\n"
                            f"Line 1 parametric eq: ${parametric_eq1}$\n\n"
                            f"Line 1 canonical eq: ${canonical_eq1}$\n\n"
                            f"Line 2 general eq: ${general_eq2}$\n\n"
                            f"Line 2 parametric eq: ${parametric_eq2}$\n\n"
                            f"Line 2 canonical eq: ${canonical_eq2}$\n\n\n",
                        )
                    )
        results.sort(key=lambda x: x[0])
        result_str = "\n".join([result[1] for result in results])
    
        if txt:
            with open("lines.txt", "w", encoding="utf-8") as file:
                file.write(result_str)
            
            
         
