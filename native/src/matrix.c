#include <include/matrix.h>

Matrix* matrix_new(int rows, int cols, MatrixType type) {
  Matrix* m = malloc(sizeof(Matrix));
  m->rows = rows;
  m->cols = cols;
  m->type = type;
  m->returned_to_js = 0;
  switch (type) {
    case TYPE_U32:
    case TYPE_I32:
    case TYPE_F32:
      // All of these types are 4 byte sized
      m->data = malloc(4 * rows * cols);
      memset(m->data, 0, 4 * rows * cols);
      break;
    default:
      free(m);
      return NULL;
  }
  return m;
}

Matrix* matrix_new_randf(int rows, int cols) {
  Matrix* m = malloc(sizeof(Matrix));
  m->rows = rows;
  m->cols = cols;
  m->type = TYPE_F32;
  m->returned_to_js = 0;
  float* data = malloc(4 * rows * cols);
  for (int i = 0; i < rows * cols; i++) {
    data[i] = gauss_rand();
  }
  m->data = (void*)data;
  return m;
}

Matrix* matrix_new_from_array(int rows, int cols, MatrixType type, void* data) {
  Matrix* m = matrix_new(rows, cols, type);
  if (m == NULL) {
    return NULL;
  }
  switch (type) {
    case TYPE_U32:
    case TYPE_I32:
    case TYPE_F32:
      memcpy(m->data, data, 4 * rows * cols);
      break;
    default:
      matrix_free(m);
      return NULL;
  }
  return m;
}

Matrix* matrix_new_from_array_zero_copy(int rows, int cols, MatrixType type, void* data) {
  Matrix* m = malloc(sizeof(Matrix));
  m->rows = rows;
  m->cols = cols;
  m->type = type;
  m->data = data;
  m->returned_to_js = 0;
  return m;
}

Matrix* matrix_new_fill_u32(int rows, int cols, uint32_t v) {
  Matrix* m = matrix_new(rows, cols, TYPE_U32);
  uint32_t* data = (uint32_t*) m->data;
  for (int i = 0; i < rows * cols; i++) {
    data[i] = v;
  }
  return m;
}

Matrix* matrix_new_fill_i32(int rows, int cols, int32_t v) {
  Matrix* m = matrix_new(rows, cols, TYPE_I32);
  int32_t* data = (int32_t*) m->data;
  for (int i = 0; i < rows * cols; i++) {
    data[i] = v;
  }
  return m;
}

Matrix* matrix_new_fill_f32(int rows, int cols, float v) {
  Matrix* m = matrix_new(rows, cols, TYPE_F32);
  float* data = (float*) m->data;
  for (int i = 0; i < rows * cols; i++) {
    data[i] = v;
  }
  return m;
}

Matrix* matrix_copy(Matrix* m) {
  Matrix* copy = matrix_new(m->rows, m->cols, m->type);
  switch (m->type) {
    case TYPE_U32:
    case TYPE_I32:
    case TYPE_F32:
      memcpy(copy->data, m->data, 4 * m->rows * m->cols);
      break;
    default:
      matrix_free(copy);
      return NULL;
  }
  return copy;
}

Matrix* matrix_map_f32(Matrix* m, float (*f) (float)) {
  if (m->type != TYPE_F32) return NULL;
  Matrix* result = matrix_new(m->rows, m->cols, m->type);
  float* resultData = result->data;
  float* mData = m->data;
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      resultData[i * m->cols + j] = f(mData[i * m->cols + j]);
    }
  }
  return result;
}

Matrix* matrix_map_i32(Matrix* m, int32_t (*f) (int32_t)) {
  if (m->type != TYPE_I32) return NULL;
  Matrix* result = matrix_new(m->rows, m->cols, m->type);
  int32_t* resultData = result->data;
  int32_t* mData = m->data;
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      resultData[i * m->cols + j] = f(mData[i * m->cols + j]);
    }
  }
  return result;
}

Matrix* matrix_map_u32(Matrix* m, uint32_t (*f) (uint32_t)) {
  if (m->type != TYPE_U32) return NULL;
  Matrix* result = matrix_new(m->rows, m->cols, m->type);
  uint32_t* resultData = result->data;
  uint32_t* mData = m->data;
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      resultData[i * m->cols + j] = f(mData[i * m->cols + j]);
    }
  }
  return result;
}

Matrix* matrix_dot(Matrix* a, Matrix* b, Matrix* result) {
  if (a->type != b->type) return NULL;
  if (a->cols != b->rows) {
    printf("Matrix dimensions do not match for dot product\n");
    return NULL;
  }
  
  if (result == NULL) result = matrix_new(a->rows, b->cols, a->type);
  else {
    if (result->type != a->type) return NULL;
    if (result->rows != a->rows) return NULL;
    if (result->cols != b->cols) return NULL;
  }

  switch (a->type) {
    case TYPE_U32: {
      uint32_t* aData = a->data;
      uint32_t* bData = b->data;
      uint32_t* resultData = result->data;
      int i, j, k;
      #pragma omp parallel for private(i, j, k) shared(aData, bData, resultData) 
      for (i = 0; i < a->rows; i++) {
        for (j = 0; j < b->cols; j++) {
          uint32_t sum = 0;
          for (k = 0; k < a->cols; k++) {
            sum += aData[i * a->cols + k] * bData[k * b->cols + j];
          }
          resultData[i * b->cols + j] = sum;
        }
      }
      break;
    }

    case TYPE_I32: {
      int32_t* aData = a->data;
      int32_t* bData = b->data;
      int32_t* resultData = result->data;
      int i, j, k;
      #pragma omp parallel for private(i, j, k) shared(aData, bData, resultData) 
      for (i = 0; i < a->rows; i++) {
        for (j = 0; j < b->cols; j++) {
          int32_t sum = 0;
          for (k = 0; k < a->cols; k++) {
            sum += aData[i * a->cols + k] * bData[k * b->cols + j];
          }
          resultData[i * b->cols + j] = sum;
        }
      }
      break;
    }

    case TYPE_F32: {
      float* aData = a->data;
      float* bData = b->data;
      float* resultData = result->data;
      int i, j, k;
      #pragma omp parallel for private(i, j, k) shared(aData, bData, resultData) 
      for (i = 0; i < a->rows; i++) {
        for (j = 0; j < b->cols; j++) {
          float sum = 0;
          for (k = 0; k < a->cols; k++) {
            sum += aData[i * a->cols + k] * bData[k * b->cols + j];
          }
          resultData[i * b->cols + j] = sum;
        }
      }
      break;
    }

    default:
      matrix_free(result);
      return NULL;
  }
  return result;
}

Matrix* matrix_add(Matrix* a, Matrix* b, Matrix* result) {
  if (a->type != b->type) return NULL;
  if (a->rows != b->rows || a->cols != b->cols) return NULL;
  
  if (result == NULL) result = matrix_new(a->rows, a->cols, a->type);
  else {
    if (result->type != a->type) return NULL;
    if (result->rows != a->rows) return NULL;
    if (result->cols != a->cols) return NULL;
  }

  switch (a->type) {
    case TYPE_U32: {
      uint32_t* aData = a->data;
      uint32_t* bData = b->data;
      uint32_t* resultData = result->data;
      for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
          resultData[i * a->cols + j] = aData[i * a->cols + j] + bData[i * a->cols + j];
        }
      }
      break;
    }

    case TYPE_I32: {
      int32_t* aData = a->data;
      int32_t* bData = b->data;
      int32_t* resultData = result->data;
      for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
          resultData[i * a->cols + j] = aData[i * a->cols + j] + bData[i * a->cols + j];
        }
      }
      break;
    }

    case TYPE_F32: {
      float* aData = a->data;
      float* bData = b->data;
      float* resultData = result->data;
      for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
          resultData[i * a->cols + j] = aData[i * a->cols + j] + bData[i * a->cols + j];
        }
      }
      break;
    }

    default:
      matrix_free(result);
      return NULL;
  }
  return result;
}

Matrix* matrix_sub(Matrix* a, Matrix* b, Matrix* result) {
  if (a->type != b->type) return NULL;
  if (a->rows != b->rows || a->cols != b->cols) return NULL;
  
  if (result == NULL) result = matrix_new(a->rows, a->cols, a->type);
  else {
    if (result->type != a->type) return NULL;
    if (result->rows != a->rows) return NULL;
    if (result->cols != a->cols) return NULL;
  }

  switch (a->type) {
    case TYPE_U32: {
      uint32_t* aData = a->data;
      uint32_t* bData = b->data;
      uint32_t* resultData = result->data;
      for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
          resultData[i * a->cols + j] = aData[i * a->cols + j] - bData[i * a->cols + j];
        }
      }
      break;
    }

    case TYPE_I32: {
      int32_t* aData = a->data;
      int32_t* bData = b->data;
      int32_t* resultData = result->data;
      for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
          resultData[i * a->cols + j] = aData[i * a->cols + j] - bData[i * a->cols + j];
        }
      }
      break;
    }

    case TYPE_F32: {
      float* aData = a->data;
      float* bData = b->data;
      float* resultData = result->data;
      for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
          resultData[i * a->cols + j] = aData[i * a->cols + j] - bData[i * a->cols + j];
        }
      }
      break;
    }

    default:
      matrix_free(result);
      return NULL;
  }
  return result;
}

Matrix* matrix_add_f32(Matrix* a, float b) {
  if (a->type != TYPE_F32) return NULL;
  
  Matrix* result = matrix_new(a->rows, a->cols, a->type);
  float* aData = a->data;
  float* resultData = result->data;
  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->cols; j++) {
      resultData[i * a->cols + j] = aData[i * a->cols + j] + b;
    }
  }
  return result;
}

Matrix* matrix_sub_f32(Matrix* a, float b) {
  if (a->type != TYPE_F32) return NULL;
  
  Matrix* result = matrix_new(a->rows, a->cols, a->type);
  float* aData = a->data;
  float* resultData = result->data;
  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->cols; j++) {
      resultData[i * a->cols + j] = aData[i * a->cols + j] - b;
    }
  }
  return result;
}

Matrix* matrix_mul_f32(Matrix* a, float b) {
  if (a->type != TYPE_F32) return NULL;
  
  Matrix* result = matrix_new(a->rows, a->cols, a->type);
  float* aData = a->data;
  float* resultData = result->data;
  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->cols; j++) {
      resultData[i * a->cols + j] = aData[i * a->cols + j] * b;
    }
  }
  return result;
}

Matrix* matrix_div_f32(Matrix* a, float b) {
  if (a->type != TYPE_F32) return NULL;
  
  Matrix* result = matrix_new(a->rows, a->cols, a->type);
  float* aData = a->data;
  float* resultData = result->data;
  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->cols; j++) {
      resultData[i * a->cols + j] = aData[i * a->cols + j] / b;
    }
  }
  return result;
}

Matrix* matrix_add_u32(Matrix* a, uint32_t b) {
  if (a->type != TYPE_U32) return NULL;
  
  Matrix* result = matrix_new(a->rows, a->cols, a->type);
  uint32_t* aData = a->data;
  uint32_t* resultData = result->data;
  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->cols; j++) {
      resultData[i * a->cols + j] = aData[i * a->cols + j] + b;
    }
  }
  return result;
}

Matrix* matrix_sub_u32(Matrix* a, uint32_t b) {
  if (a->type != TYPE_U32) return NULL;
  
  Matrix* result = matrix_new(a->rows, a->cols, a->type);
  uint32_t* aData = a->data;
  uint32_t* resultData = result->data;
  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->cols; j++) {
      resultData[i * a->cols + j] = aData[i * a->cols + j] - b;
    }
  }
  return result;
}

Matrix* matrix_mul_u32(Matrix* a, uint32_t b) {
  if (a->type != TYPE_U32) return NULL;
  
  Matrix* result = matrix_new(a->rows, a->cols, a->type);
  uint32_t* aData = a->data;
  uint32_t* resultData = result->data;
  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->cols; j++) {
      resultData[i * a->cols + j] = aData[i * a->cols + j] * b;
    }
  }
  return result;
}

Matrix* matrix_div_u32(Matrix* a, uint32_t b) {
  if (a->type != TYPE_U32) return NULL;
  
  Matrix* result = matrix_new(a->rows, a->cols, a->type);
  uint32_t* aData = a->data;
  uint32_t* resultData = result->data;
  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->cols; j++) {
      resultData[i * a->cols + j] = aData[i * a->cols + j] / b;
    }
  }
  return result;
}

Matrix* matrix_add_i32(Matrix* a, int32_t b) {
  if (a->type != TYPE_I32) return NULL;
  
  Matrix* result = matrix_new(a->rows, a->cols, a->type);
  int32_t* aData = a->data;
  int32_t* resultData = result->data;
  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->cols; j++) {
      resultData[i * a->cols + j] = aData[i * a->cols + j] + b;
    }
  }
  return result;
}

Matrix* matrix_sub_i32(Matrix* a, int32_t b) {
  if (a->type != TYPE_I32) return NULL;
  
  Matrix* result = matrix_new(a->rows, a->cols, a->type);
  int32_t* aData = a->data;
  int32_t* resultData = result->data;
  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->cols; j++) {
      resultData[i * a->cols + j] = aData[i * a->cols + j] - b;
    }
  }
  return result;
}

Matrix* matrix_mul_i32(Matrix* a, int32_t b) {
  if (a->type != TYPE_I32) return NULL;
  
  Matrix* result = matrix_new(a->rows, a->cols, a->type);
  int32_t* aData = a->data;
  int32_t* resultData = result->data;
  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->cols; j++) {
      resultData[i * a->cols + j] = aData[i * a->cols + j] * b;
    }
  }
  return result;
}

Matrix* matrix_div_i32(Matrix* a, int32_t b) {
  if (a->type != TYPE_I32) return NULL;
  
  Matrix* result = matrix_new(a->rows, a->cols, a->type);
  int32_t* aData = a->data;
  int32_t* resultData = result->data;
  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->cols; j++) {
      resultData[i * a->cols + j] = aData[i * a->cols + j] / b;
    }
  }
  return result;
}

Matrix* matrix_transpose(Matrix* m) {
  Matrix* result = matrix_new(m->cols, m->rows, m->type);
  switch (m->type) {
    case TYPE_U32: {
      uint32_t* mData = m->data;
      uint32_t* resultData = result->data;
      for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
          resultData[j * m->rows + i] = mData[i * m->cols + j];
        }
      }
      break;
    }

    case TYPE_I32: {
      int32_t* mData = m->data;
      int32_t* resultData = result->data;
      for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
          resultData[j * m->rows + i] = mData[i * m->cols + j];
        }
      }
      break;
    }

    case TYPE_F32: {
      float* mData = m->data;
      float* resultData = result->data;
      for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
          resultData[j * m->rows + i] = mData[i * m->cols + j];
        }
      }
      break;
    }

    default:
      matrix_free(result);
      return NULL;
  }
  return result;
}

void matrix_print(Matrix* m, char* name) {
  printf("%s = [\n", name);
  for (int i = 0; i < m->rows; i++) {
    printf("  ");
    for (int j = 0; j < m->cols; j++) {
      switch (m->type) {
        case TYPE_U32:
          printf("%u", ((uint32_t*)m->data)[i * m->cols + j]);
          break;
        case TYPE_I32:
          printf("%d", ((int32_t*)m->data)[i * m->cols + j]);
          break;
        case TYPE_F32:
          printf("%f", ((float*)m->data)[i * m->cols + j]);
          break;
        default:
          printf("?");
          break;
      }
      if (j < m->cols - 1) printf(", ");
    }
    printf("\n");
  }
  printf("]\n");
}

void matrix_serialize(Matrix* m, FILE* f) {
  fwrite(&m->type, sizeof(MatrixType), 1, f);
  fwrite(&m->rows, sizeof(uint32_t), 1, f);
  fwrite(&m->cols, sizeof(uint32_t), 1, f);
  fwrite(m->data, 4, m->rows * m->cols, f);
}

Matrix* matrix_deserialize(FILE* f) {
  MatrixType type;
  uint32_t rows, cols;
  fread(&type, sizeof(MatrixType), 1, f);
  fread(&rows, sizeof(uint32_t), 1, f);
  fread(&cols, sizeof(uint32_t), 1, f);
  Matrix* m = matrix_new(rows, cols, type);
  fread(m->data, 4, rows * cols, f);
  return m;
}

void matrix_free(Matrix* m) {
  CHECK_NULL(m);
  if (m->data != NULL) free(m->data);
  free(m);
}
