import pytest
from dash.exceptions import PreventUpdate

import app as app_module
import pages.predict as predict_page

VALID_INPUT = dict(
    brand=None,
    year=2015,
    km_driven=50000,
    fuel=None,
    seller_type=None,
    transmission=None,
    owner=1,
    mileage=20.0,
    engine=1200.0,
    max_power=82.0,
    seats=5.0,
)


def call_predict(n_clicks=1, **overrides):
    values = dict(VALID_INPUT)
    values.update(overrides)
    return predict_page.predict_price(n_clicks, *values.values())


def rendered_text(component):
    """Flatten a Dash component tree down to its text content."""
    if isinstance(component, str):
        return component
    if isinstance(component, (list, tuple)):
        return " ".join(rendered_text(child) for child in component)
    children = getattr(component, "children", None)
    return rendered_text(children) if children is not None else ""


def test_app_exposes_a_wsgi_server():
    assert app_module.server is not None


def test_app_registers_home_and_predict_pages():
    import dash

    paths = {page["path"] for page in dash.page_registry.values()}
    assert {"/", "/predict"} <= paths


def test_predict_page_has_a_layout():
    assert predict_page.layout is not None


def test_no_click_does_not_update():
    with pytest.raises(PreventUpdate):
        call_predict(n_clicks=0)


@pytest.mark.parametrize(
    "missing, expected",
    [
        ({"year": None}, "year"),
        ({"max_power": None}, "max power"),
    ],
)
def test_missing_required_field_returns_a_warning(missing, expected):
    result = call_predict(**missing)
    assert expected in rendered_text(result)


def test_missing_both_required_fields_names_both():
    text = rendered_text(call_predict(year=None, max_power=None))
    assert "year" in text and "max power" in text


def test_valid_input_returns_a_formatted_price():
    text = rendered_text(call_predict())
    assert "Estimated selling price" in text
    price = text.split("Estimated selling price")[-1].strip()
    assert price.replace(",", "").isdigit()
    assert int(price.replace(",", "")) > 0
