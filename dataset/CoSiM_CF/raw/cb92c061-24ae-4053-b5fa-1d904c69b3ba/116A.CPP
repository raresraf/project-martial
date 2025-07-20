#include <iostream>
using namespace std;

int main() {
  int currently = 0;
  int total = 0;
  int n;
  cin >> n;
  int a, b;
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    currently -= a;
    currently += b;
    if (currently > total)
      total = currently;
  }
  cout << total << "\n";
}
