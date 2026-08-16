from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    MedInc: float = Field(
        gt=0,
        description="Median income in tens of thousands of dollars"
    )

    HouseAge: float = Field(
        gt=0,
        le=100,
        description="Average house age in years"
    )

    AveRooms: float = Field(
        gt=0,
        le=150,
        description="Average number of rooms per household"
    )

    AveBedrms: float = Field(
        gt=0,
        le=50,
        description="Average number of bedrooms per household"
    )

    Population: float = Field(
        gt=0,
        description="Block group population"
    )

    AveOccup: float = Field(
        gt=0,
        le=100,
        description="Average number of household members"
    )

    Latitude: float = Field(
        ge=32,
        le=42,
        description="Block group latitude"
    )

    Longitude: float = Field(
        ge=-125,
        le=-114,
        description="Block group longitude"
    )