#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  int a = 0, p, q, k = 0;
  for (int i = 0; i < n; i++) {
    cin >> p >> q;
    k -= p;
    k += q;
    if (k > a)
      a = k;
  }
  cout << a << endl;
}