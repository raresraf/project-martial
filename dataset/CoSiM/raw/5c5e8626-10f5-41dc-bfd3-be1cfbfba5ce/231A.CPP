#include <iostream>
using namespace std;

int main() {
  int n, count = 0, f1, f2, f3;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> f1 >> f2 >> f3;
    if (f1 + f2 + f3 >= 2)
      count++;
  }
  cout << count << endl;
  return 0;
}
