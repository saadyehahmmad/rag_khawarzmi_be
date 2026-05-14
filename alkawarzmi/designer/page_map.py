"""
Survey Designer screen map for ``page_id`` values sent by the Angular app (KhwarzmiDesigner).

Package: ``alkawarzmi.designer``. Aligned with ``vector_stores/designer/ROUTES.md``:
- ``None``, ``""``, or whitespace-only ``page_id`` → **Dashboard** (`/`).
- Other values resolve to route path + bilingual title + short description.

Used for user-facing guidance (payload prescripts) and LLM context (answer prompt).
"""

from __future__ import annotations

import re
from typing import TypedDict


class DesignerPageInfo(TypedDict):
    """Resolved Designer location for prompts and UI copy."""

    canonical: str
    path: str
    title_en: str
    title_ar: str
    desc_en: str
    desc_ar: str


# Canonical slug → (Angular route path, EN title, AR title, EN desc, AR desc)
# Source: vector_stores/designer/ROUTES.md (Main Application Routes)
_DESIGNER_PAGES: dict[str, tuple[str, str, str, str, str]] = {
    "dashboard": (
        "/",
        "Dashboard",
        "لوحة التحكم",
        "Overview of all surveys, statistics, and recent activity.",
        "نظرة عامة على جميع الاستبيانات والإحصاءات والنشاط الأخير.",
    ),
    "home": (
        "/home",
        "My Surveys",
        "استبياناتي",
        "List of surveys belonging to the current user, with quick actions.",
        "قائمة الاستبيانات التابعة للمستخدم الحالي مع الإجراءات السريعة.",
    ),
    "builder": (
        "/builder",
        "Survey Builder",
        "منشئ الاستبيان",
        "Design and manage survey questions, sections, and structure.",
        "تصميم أسئلة الاستبيان وأقسامه وهيكله وإدارتها.",
    ),
    "design": (
        "/design",
        "Survey Design",
        "تصميم الاستبيان",
        "Customize the visual appearance and theme of the survey.",
        "تخصيص المظهر المرئي وسمة الاستبيان.",
    ),
    "config": (
        "/config",
        "Survey Configuration",
        "إعداد الاستبيان",
        "Configure survey settings, behavior, and AI-assisted options.",
        "ضبط إعدادات الاستبيان وسلوكه والخيارات المدعومة بالذكاء الاصطناعي.",
    ),
    "logicpage": (
        "/logicPage",
        "Survey Logic",
        "منطق الاستبيان",
        "Define conditional branching and skip-logic rules for the survey.",
        "تحديد قواعد التفرع الشرطي والتخطي في الاستبيان.",
    ),
    "logicpage2": (
        "/logicPage2",
        "Advanced Logic",
        "المنطق المتقدم",
        "Legacy logic editor for complex survey branching rules.",
        "محرر المنطق القديم لقواعد التفرع المعقدة في الاستبيان.",
    ),
    "publish": (
        "/publish",
        "Publish Survey",
        "نشر الاستبيان",
        "Review and publish the survey to make it accessible to respondents.",
        "مراجعة الاستبيان ونشره لإتاحته للمستجيبين.",
    ),
    "createnew": (
        "/createNew",
        "Create New Survey",
        "إنشاء استبيان جديد",
        "Start a new survey by choosing a template or blank form.",
        "بدء استبيان جديد باختيار قالب أو نموذج فارغ.",
    ),
    "reports": (
        "/reports/:id",
        "Survey Reports",
        "تقارير الاستبيان",
        "View analytical reports and charts for a specific survey.",
        "عرض التقارير التحليلية والرسوم البيانية لاستبيان محدد.",
    ),
    "sharedreports": (
        "/shared-reports/:id",
        "Shared Reports",
        "التقارير المشتركة",
        "Access a publicly shared report view without authentication.",
        "الوصول إلى تقرير مشترك بشكل عام دون الحاجة إلى مصادقة.",
    ),
    "usermanagement": (
        "/userManagement",
        "User Management",
        "إدارة المستخدمين",
        "Browse and manage all platform users and their information.",
        "استعراض جميع مستخدمي المنصة وإدارة معلوماتهم.",
    ),
    "groups": (
        "/groups",
        "Groups",
        "المجموعات",
        "View and manage all respondent groups in the system.",
        "عرض مجموعات المستجيبين في النظام وإدارتها.",
    ),
    "assignsurveys": (
        "/assign-surveys",
        "Assign Surveys",
        "تعيين الاستبيانات",
        "Assign or unassign surveys to specific users or groups.",
        "تعيين الاستبيانات أو إلغاء تعيينها لمستخدمين أو مجموعات محددة.",
    ),
    "rolespermission": (
        "/roles-permission",
        "Roles & Permissions",
        "الأدوار والصلاحيات",
        "Manage platform roles and the permissions assigned to each.",
        "إدارة أدوار المنصة والصلاحيات المعينة لكل منها.",
    ),
    "organization": (
        "/organization",
        "Organization",
        "المنظمة",
        "View and manage organization details, industries, and types.",
        "عرض تفاصيل المنظمة والصناعات والأنواع وإدارتها.",
    ),
    "calendar": (
        "/calendarSetup/calendar",
        "Calendar Setup",
        "إعداد التقويم",
        "Configure date schedules and event timelines for surveys.",
        "ضبط جداول التواريخ والجداول الزمنية للأحداث المرتبطة بالاستبيانات.",
    ),
    "calendarsetup": (
        "/calendarSetup/calendar",
        "Calendar Setup",
        "إعداد التقويم",
        "Configure date schedules and event timelines for surveys.",
        "ضبط جداول التواريخ والجداول الزمنية للأحداث المرتبطة بالاستبيانات.",
    ),
    "codinglist": (
        "/coding-list",
        "Coding List",
        "قائمة الترميز",
        "Manage predefined code lists used across survey questions.",
        "إدارة قوائم الرموز المحددة مسبقاً والمستخدمة في أسئلة الاستبيانات.",
    ),
    "surveydataconsumption": (
        "/survey-data-consumption",
        "Data Consumption",
        "استهلاك بيانات الاستبيان",
        "Monitor and analyze data collected from published surveys.",
        "مراقبة وتحليل البيانات المجمعة من الاستبيانات المنشورة.",
    ),
    "qbank": (
        "/qbank",
        "Question Bank",
        "بنك الأسئلة",
        "Browse and manage the shared library of reusable survey questions.",
        "استعراض المكتبة المشتركة من الأسئلة القابلة لإعادة الاستخدام وإدارتها.",
    ),
    "uploadsamples": (
        "/upload-samples",
        "Upload Samples",
        "رفع العينات",
        "View and manage all uploaded respondent sample files.",
        "عرض ملفات عينات المستجيبين المرفوعة وإدارتها.",
    ),
    "rulelogs": (
        "/ruleLogs",
        "Rule Logs",
        "سجلات القواعد",
        "Inspect execution logs and history of automation rule runs.",
        "فحص سجلات التنفيذ وتاريخ تشغيل قواعد الأتمتة.",
    ),
    "dataquality": (
        "/data-quality",
        "Data Quality",
        "جودة البيانات",
        "Review and validate the quality of collected survey responses.",
        "مراجعة جودة البيانات المجمعة من استجابات الاستبيان والتحقق منها.",
    ),
    "establishmentsframe": (
        "/establishments-frame",
        "Establishments Frame",
        "إطار المنشآت",
        "Manage and configure the organizational establishments framework.",
        "إدارة إطار المنشآت التنظيمية وتهيئته.",
    ),
    "registeredprofiles": (
        "/registered-profiles",
        "Registered Profiles",
        "الملفات المسجلة",
        "View and manage respondent profiles registered in the system.",
        "عرض ملفات المستجيبين المسجلة في النظام وإدارتها.",
    ),
    "account": (
        "/account",
        "My Profile",
        "ملفي الشخصي",
        "View and update personal account information and preferences.",
        "عرض معلومات الحساب الشخصي وتفضيلاته وتحديثها.",
    ),
}


def _alnum_key(s: str) -> str:
    """Lowercase alphanumeric key for fuzzy alias matching."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


# Non-trivial aliases only — hyphenated/underscore variants are handled by _alnum_key.
_PAGE_ID_ALIASES: dict[str, str] = {
    "root":            "dashboard",
    "index":           "dashboard",
    "main":            "dashboard",
    "homepage":        "home",   # also catches "home_page" after _alnum_key strips "_"
    "mysurveys":       "home",
    "createnewsurvey": "createnew",
    "myaccount":       "account",
}


def resolve_designer_page(page_id: str | None) -> DesignerPageInfo:
    """
    Resolve ``page_id`` to route metadata.

    Empty or missing ``page_id`` is treated as **Dashboard** (`/`), per ROUTES.md.

    Args:
        page_id: Value from the chat request (may be ``""``).

    Returns:
        ``DesignerPageInfo``; unknown non-empty ids still return a sensible default
        using the raw segment as title fallback.
    """
    raw = (str(page_id) if page_id is not None else "").strip()
    if not raw:
        canonical = "dashboard"
    else:
        key = _alnum_key(raw)
        canonical = _PAGE_ID_ALIASES.get(key, key)

    row = _DESIGNER_PAGES.get(canonical)
    if row is not None:
        path, title_en, title_ar, desc_en, desc_ar = row
        return {
            "canonical": canonical,
            "path": path,
            "title_en": title_en,
            "title_ar": title_ar,
            "desc_en": desc_en,
            "desc_ar": desc_ar,
        }

    # Unknown custom page_id — still show path-like hint
    slug = raw if raw else "(empty)"
    return {
        "canonical": canonical,
        "path": f"/{slug}" if raw and not raw.startswith("/") else (raw or "/"),
        "title_en": f"Designer screen ({slug})",
        "title_ar": f"شاشة المصمّم ({slug})",
        "desc_en": "Screen not listed in the standard Designer route map; treat as a secondary or custom view.",
        "desc_ar": "شاشة غير مدرجة في خريطة مسارات المصمّم القياسية؛ اعتبرها عرضاً ثانوياً أو مخصصاً.",
    }


def resolved_designer_route_is_known(info: DesignerPageInfo) -> bool:
    """True when ``page_id`` mapped to a route defined in ``_DESIGNER_PAGES`` (not the generic unknown stub)."""
    return info["canonical"] in _DESIGNER_PAGES


def describe_designer_location_for_user(page_id: str | None, *, lang: str) -> str:
    """
    Markdown block for the “where am I” prescript: product context, screen title,
    what the screen is for, and the Angular route for power users.
    """
    info = resolve_designer_page(page_id)
    ar = lang == "ar"
    title = info["title_ar"] if ar else info["title_en"]
    desc = info["desc_ar"] if ar else info["desc_en"]
    path = info["path"]
    if ar:
        return (
            "## أين أنت الآن\n"
            "أنت داخل **مصمّم استبيانات الخوارزمي** — التطبيق الذي تبني فيه الاستبيانات، "
            "المنطق، التصميم، والنشر قبل أن يصل الاستبيان للميدان أو الاتصال.\n\n"
            "### الشاشة الحالية\n"
            f"**{title}**\n\n"
            f"{desc}\n\n"
        )
    return (
        "## Where you are\n"
        "You're in **Al-Khwarzmi Survey Designer** — the workspace where you create surveys, "
        "set **logic / skip rules**, adjust **look & feel**, and **publish** before field teams or call center staff use them.\n\n"
        "### Current screen\n"
        f"**{title}**\n\n"
        f"{desc}\n\n"
    )


def describe_designer_location_for_prompt(page_id: str | None, *, lang: str) -> str:
    """One paragraph for the answer LLM: explicit Designer location + description."""
    info = resolve_designer_page(page_id)
    ar = lang == "ar"
    title = info["title_ar"] if ar else info["title_en"]
    desc  = info["desc_ar"]  if ar else info["desc_en"]
    if ar:
        return (
            "**مكان المستخدم في مصمّم الاستبيانات (KhwarzmiDesigner):** "
            f"{title} — المسار {info['path']}. {desc}\n\n"
        )
    return (
        "**User location in Survey Designer (KhwarzmiDesigner):** "
        f"{title} — route {info['path']}. {desc}\n\n"
    )
