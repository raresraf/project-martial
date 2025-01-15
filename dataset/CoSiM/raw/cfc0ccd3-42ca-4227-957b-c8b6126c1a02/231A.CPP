// Accepted
// 231A - Team
#include <iostream>
using namespace std;
int main() {
  int n, a, b, c, agree = 0, total = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    agree = 0;
    cin >> a >> b >> c;
    if (a == 1) {
      agree++;
    }
    if (b == 1) {
      agree++;
    }
    if (c == 1) {
      agree++;
    }
    // cout<<agree<<endl;
    if (agree >= 2) {
      total++;
    }
  }
  cout << total << endl;
}
