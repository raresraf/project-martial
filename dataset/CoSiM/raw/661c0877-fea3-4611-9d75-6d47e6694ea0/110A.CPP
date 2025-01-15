#include <iostream>

using namespace std;

int main() {
  int ans = 0;
  long long n;
  cin >> n;
  while (n > 0ll) {
    if (n % 10ll == 4ll || n % 10ll == 7ll) {
      ++ans;
    }
    n /= 10ll;
  }
  cout << (ans == 4 || ans == 7 ? "YES" : "NO") << endl;
  return 0;
}
