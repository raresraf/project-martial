#include <cmath>
#include <iostream>
using namespace std;

int main() {
  int l;
  cin >> l;

  int A[1000], B[1000];
  for (int i = 0; i < l; i++) {
    cin >> A[i] >> B[i];
  }

  int a = 0, b = 0;
  for (int i = 0; i < l; i++) {
    a = (a - A[i]) + B[i];
    b = max(a, b);
  }

  cout << b << endl;
}
