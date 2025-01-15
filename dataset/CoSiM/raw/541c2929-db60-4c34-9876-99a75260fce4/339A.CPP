#include <iostream>
#include <string>
using namespace std;

void BBsort(int *A, int size) {
  for (int i = 0; i < size - 1; i++)
    for (int j = 0; j < size - 1; j++)
      if (A[j] > A[j + 1]) {
        int temp = A[j];
        A[j] = A[j + 1];
        A[j + 1] = temp;
      }
}

int main() {

  string op;
  cin >> op;
  int size = op.length() / 2 + 1;

  int *arr = new int[size];

  for (int i = 0; i < op.length(); i += 2) {
    arr[i / 2] = op[i] - 48;
  }

  BBsort(arr, size);

  for (int i = 0; i < size; i++) {
    cout << arr[i];
    if (i != size - 1)
      cout << "+";
  }
  cout << "\n";
  return 0;
}
