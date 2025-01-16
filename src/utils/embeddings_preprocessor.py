def preprocess_profile_text(profile):
    """
    Preprocess a single profile to create a text representation.
    Combines 'Summary', 'Skills', and 'Experience' fields into a single text block.
    """
    summary = profile.get("Summary", "")
    skills = " ".join(profile.get("Skills", []))
    experiences = " ".join(exp.get("description", "") for exp in profile.get("Experience", []))
    return f"{summary} {skills} {experiences}"

def preprocess_profiles(profiles):
    """
    Apply text preprocessing to all profiles.
    """
    return {cid: preprocess_profile_text(profile) for cid, profile in profiles.items()}
