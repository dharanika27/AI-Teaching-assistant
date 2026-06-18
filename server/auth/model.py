from pydantic import BaseModel, ConfigDict, Field


class StudentUser(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # id:int
    fullname:str = Field(alias="fullName")
    email:str
    username:str = Field(alias="userName")
    password:str
    grade:int
    school:str


class TeacherUser(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # id:int
    fullname:str = Field(alias="fullName")
    email:str
    username:str = Field(alias="userName")
    password:str
    school:str
