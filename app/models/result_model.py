from sqlalchemy import Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base

# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)

    history_id = Column(Integer, ForeignKey("histories.id"), nullable=False)

    x1 = Column(Float)
    y1 = Column(Float)
    x2 = Column(Float)
    y2 = Column(Float)

    detection_score = Column(Float)

    variety = Column(String)
    variety_score = Column(Float)

    maturity = Column(String)
    maturity_score = Column(Float)

    history = relationship("History", back_populates="results")