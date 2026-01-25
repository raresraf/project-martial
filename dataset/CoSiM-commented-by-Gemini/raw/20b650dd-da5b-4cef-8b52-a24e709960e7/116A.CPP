#include <iostream>
using namespace std;
int main() {
  int n, in, out, ans = 0, max = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> out >> in;
    ans = ans - out + in;
    max = ans > max ? ans : max;
  }
  cout << max << endl;

  return 0;
}
