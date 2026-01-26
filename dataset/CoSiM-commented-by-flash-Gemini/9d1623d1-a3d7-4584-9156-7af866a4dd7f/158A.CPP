#include <iostream>

using namespace std;

int main() {
  int n, k, c = 0;
  cin >> n >> k;
  int participantes[n];
  for (int i = 1; i <= n; i++) {
    cin >> participantes[i];
  }
  for (int i = 1; i <= n; i++) {
    if (participantes[i] >= participantes[k]) {
      if (participantes[i] != 0)
        c++;
    }
  }
  cout << c;
  return 0;
}
