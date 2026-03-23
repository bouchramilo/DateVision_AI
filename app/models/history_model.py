from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base
from datetime import datetime



# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
class History(Base):
    __tablename__ = "histories"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    image = Column(String, nullable=False)
    report = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # 🔹 relation
    user = relationship("User", back_populates="histories")
    results = relationship("Result", back_populates="history", cascade="all, delete")