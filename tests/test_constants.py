from nird import constants


def test_conversion_miles_to_kilometres_roundtrip():
    """Test the conversion from miles to kilometres."""
    initial_miles = 10
    kilometers = initial_miles * constants.CONV_MILE_TO_KM
    actual_miles = kilometers * constants.CONV_KM_TO_MILE
    difference = abs(actual_miles - initial_miles)
    assert (
        difference < 1e-3
    ), f"Expected {initial_miles} miles, got {actual_miles} miles"
