#include <iostream>

using namespace std;

int main() {
  int n, ans = 0;
  cin >> n;                     // n: number of problems, ans: final answer
  for (int i = 0; i < n; i++) { // loop n times
    int x, y, z;
    cin >> x >> y >> z; // the three members
    ans += (x + y + z >= 2);
  }
  cout << ans << endl;
  return 0;
}
