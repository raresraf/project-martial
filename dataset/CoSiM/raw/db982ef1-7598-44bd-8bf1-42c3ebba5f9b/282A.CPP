#include <iostream>
using namespace std;
int main() {

  int n, x = 0;
  cin >> n;

  for (int i = 0; i < n; i++) {

    string sv;
    cin >> sv;
    sv == "X++" || sv == "++X" ? x++ : x--;
  }

  cout << x;

  return 0;
}
