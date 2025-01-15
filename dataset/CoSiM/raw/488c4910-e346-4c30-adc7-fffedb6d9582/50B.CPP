#include <iostream>
#include <vector>

using namespace std;

int main() {
  long long t;
  cin >> t;

  while (t--) {
    long long n;
    cin >> n;

    if ((n / 2) % 2 != 0) {
      cout << "NO" << endl;
    } else {
      cout << "YES" << endl;
      long long counte = 0, counto = 0;
      for (int i = 0; i < n / 2; i++) {
        cout << 2 * (i + 1) << " ";
        counte += 2 * (i + 1);
      }

      for (int i = 0; i < (n / 2) - 1; i++) {
        cout << (2 * (i + 1)) - 1 << " ";
        counto += (2 * (i + 1)) - 1;
      }
      cout << counte - counto << endl;
    }
  }

  return 0;
}
