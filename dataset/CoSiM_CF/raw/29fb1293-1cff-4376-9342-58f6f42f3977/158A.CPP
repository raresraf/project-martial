#include <iostream>

using namespace std;

int main() {
  // n the number of participants
  // n >= k
  int n = 0, k = 0;
  cin >> n >> k;
  int participants[n];
  int x = 0;

  for (int i = 0; i < n; i++) {
    cin >> participants[i];
  }

  for (int i = 0; i < n; i++) {
    if (participants[i] > k) {
      x++;
    }
  }

  cout << x << endl;
  return 0;
}
