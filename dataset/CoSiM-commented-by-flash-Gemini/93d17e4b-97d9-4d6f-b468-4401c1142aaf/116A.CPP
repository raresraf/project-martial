#include <iostream>
using namespace std;
int main() {
  int i, max(0), a, b, curr(0), n;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> a >> b;
    curr += -a + b;
    if (curr > max)
      max = curr;
  }
  cout << max << endl;
  return 0;
}
