import numpy as np
class Tensor(np.ndarray):
	"""docstring for Tensor"""
	def __new__(cls, input_array):
		#assert isinstance(input_array,np.ndarray) or isinstance(input_array,list) : "Input Array must be an numpy array or list"
		return np.asarray(input_array,dtype='f4').view(cls)

	def __array_finalize__(self, obj) -> None:
		if obj is None: return
		# This attribute should be maintained!
		default_attributes = {"attr": 1}
		self.__dict__.update(default_attributes)

'''	def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):  # this method is called whenever you use a ufunc
			args = ((i.view(np.ndarray) if isinstance(i, ArraySubclass) else i) for i in inputs)
			outputs = kwargs.pop('out', None)
			if outputs:
				kwargs['out'] = tuple((o.view(np.ndarray) if isinstance(o, ArraySubclass) else o) for o in outputs)
			else:
				outputs = (None,) * ufunc.nout
			results = super().__array_ufunc__(ufunc, method, *args, **kwargs)  # pylint: disable=no-member
			if results is NotImplemented:
				return NotImplemented
			if method == 'at':
				return
				results = (results,)
			results = tuple((self._copy_attrs_to(result) if output is None else output)
	                        for result, output in zip(results, outputs))
			return results[0] if len(results) == 1 else results
	
	def _copy_attrs_to(self, target):
		target = target.view(ArraySubclass)
		try:
			target.__dict__.update(self.__dict__)
		except AttributeError:
			pass
		return target'''