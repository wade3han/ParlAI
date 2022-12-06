import copy
import os
import random
import re
from collections import defaultdict
from typing import List, Optional, Dict, Tuple

from parlai.core.opt import Opt
from parlai.core.teachers import ParlAIDialogTeacher, create_task_agent_from_taskname
from parlai.tasks.convai2.agents import BothTeacher
from parlai.tasks.empathetic_dialogues.agents import EmpatheticDialoguesTeacher
from parlai.tasks.wizard_of_wikipedia.agents import WizardDialogKnowledgeTeacher
from parlai.utils.misc import warn_once
from parlai.utils.io import PathManager
from .build import build


##################################################
#### Teacher for the YTDIALOGUE Dataset ####
##################################################


def _processed_data_path(opt: Opt) -> str:
  # Build the data if it doesn't exist.
  build(opt)
  dt = opt['datatype'].split(':')[0]
  return os.path.join(opt['datapath'], 'ytdialogue', dt + '.txt')


class YtdialogueTeacher(ParlAIDialogTeacher):
  def __init__(self, opt, shared=None):
    opt = copy.deepcopy(opt)
    opt['parlaidialogteacher_datafile'] = _processed_data_path(opt)
    super().__init__(opt, shared)


class InteractiveTeacher(YtdialogueTeacher):
  # Dummy class to add arguments for interactive world.
  pass


class SelfchatTeacher(YtdialogueTeacher):
  # Dummy class to add arguments for interactive world.
  pass


class DefaultTeacher(YtdialogueTeacher):
  pass


def create_agents(opt):
  if not opt.get('interactive_task', False):
    return create_task_agent_from_taskname(opt)
  else:
    # interactive task has no task agents (they are attached as user agents)
    return []
