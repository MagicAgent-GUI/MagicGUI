from ..smp import *
from ..dataset import img_root_map, DATASET_TYPE
from abc import abstractmethod


class BaseModel:

    INTERLEAVE = False
    allowed_types = ['text', 'image', 'video']

    def __init__(self):
        self.dump_image_func = None

    def use_custom_prompt(self, dataset):
        """Whether to use custom prompt for the given dataset.

        Args:
            dataset (str): The name of the dataset.

        Returns:
            bool: Whether to use custom prompt. If True, will call `build_prompt` of the VLM to build the prompt.
                Default to False.
        """
        return False

    @abstractmethod
    def build_prompt(self, line, dataset):
        """Build custom prompts for a specific dataset. Called only if `use_custom_prompt` returns True.

        Args:
            line (line of pd.DataFrame): The raw input line.
            dataset (str): The name of the dataset.

        Returns:
            str: The built message.
        """
        raise NotImplementedError

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    @abstractmethod
    def generate_inner(self, message, dataset=None):
        raise NotImplementedError

    def check_content(self, msgs):
        """Check the content type of the input. Four types are allowed: str, dict, liststr, listdict.
        """
        if isinstance(msgs, str):
            return 'str'
        if isinstance(msgs, dict):
            return 'dict'
        if isinstance(msgs, list):
            types = [self.check_content(m) for m in msgs]
            if all(t == 'str' for t in types):
                return 'liststr'
            if all(t == 'dict' for t in types):
                return 'listdict'
        return 'unknown'

    def preproc_content(self, inputs):
        """Convert the raw input messages to a list of dicts.

        Args:
            inputs: raw input messages.

        Returns:
            list(dict): The preprocessed input messages. Will return None if failed to preprocess the input.
        """
        if self.check_content(inputs) == 'str':
            return [dict(type='text', value=inputs)]
        elif self.check_content(inputs) == 'dict':
            if 'type' not in inputs or  'value' not in inputs:
                raise ValueError(f"Error inputs {inputs} ,  'type' or  'value' not in it ")
            return [inputs]
        elif self.check_content(inputs) == 'liststr':
            res = []
            for s in inputs:
                mime, pth = parse_file(s)
                if mime is None or mime == 'unknown':
                    res.append(dict(type='text', value=s))
                else:
                    res.append(dict(type=mime.split('/')[0], value=pth))
            return res
        elif self.check_content(inputs) == 'listdict':
            for item in inputs:
                if 'type' not in item or 'value' not in item:
                    raise ValueError(f"Error item {item} ,  'type' or  'value' not in it ")
                mime, s = parse_file(item['value'])
                if mime is None:
                    if item['type'] != 'text':
                        raise ValueError(f"Error item {item} ,  {item['type'] } is not equal to 'text' ")
                else:
                    if mime.split('/')[0] != item['type']:
                        raise ValueError(f"Error item {item}  and mime {mime.split('/')[0] }, type not match")
                    item['value'] = s
            return inputs
        else:
            return None

    def generate(self, message, dataset=None):
        """Generate the output message.

        Args:
            message (list[dict]): The input message.
            dataset (str, optional): The name of the dataset. Defaults to None.

        Returns:
            str: The generated message.
        """
        if self.check_content(message) not in ['str', 'dict', 'liststr', 'listdict']:
            raise ValueError( f'Invalid input type: {message}')
        message = self.preproc_content(message)
        if message is  None or self.check_content(message) != 'listdict':
            raise ValueError( f'Invalid input type: {message}')
        for item in message:
            if item['type'] not in self.allowed_types:
                raise ValueError( f'Invalid input type: {item["type"]}')
        return self.generate_inner(message, dataset)

