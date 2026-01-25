#include <iostream>
#include <map>
#include <set>

using namespace std;
int main() {
  int n;
  int m;
  cin >> n >> m;
  int outputm;
  int outputn;
  if (m % 2 == 0) {
    cout << (m * n) / 2;
    return 0;
  } else {
    if (n % 2 == 0) {
      cout << (m * n) / 2;
      return 0;
    }
  }
  int modedm = m - (m % 2);
  int modedn = n - (n % 2);
  outputm = 0.5 * (modedm * n) + 0.5 * (modedn);
  outputn = 0.5 * (modedn * m) + 0.5 * (modedm);
  if (outputm > outputn) {
    cout << outputn;
    return 0;
  }
  cout << outputm;
  return 0;
}
