from __future__ import annotations

import factory
from factory.alchemy import SQLAlchemyModelFactory

from ci_app.extensions import db
from ci_app.models import Role, User


class _Session:
    @staticmethod
    def get_session():
        return db.session


class RoleFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Role
        sqlalchemy_session = db.session
        sqlalchemy_session_persistence = "flush"

    name = factory.Sequence(lambda n: f"role-{n}")
    description = factory.Faker("sentence")


class UserFactory(SQLAlchemyModelFactory):
    class Meta:
        model = User
        sqlalchemy_session = db.session
        sqlalchemy_session_persistence = "flush"

    email = factory.Sequence(lambda n: f"user{n}@example.com")
    name = factory.Faker("name")
    active = True
    password = factory.PostGeneration(lambda obj, create, extracted, **kwargs: obj.set_password("Passw0rd!"))

    @factory.post_generation
    def roles(self, create, extracted, **kwargs):  # type: ignore[override]
        if not create:
            return
        if extracted:
            for r in extracted:
                self.roles.append(r)

