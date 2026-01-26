#include <iostream>
using namespace std;
int main() {

  int n;
  int sum = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    int a, b, c;
    cin >> a >> b >> c;
    a + b + c >= 2 ? sum++ : sum;
  }

  cout << sum;

  return 0;
}
