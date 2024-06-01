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
