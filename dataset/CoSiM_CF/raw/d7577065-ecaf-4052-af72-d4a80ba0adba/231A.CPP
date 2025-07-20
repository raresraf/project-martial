#include <iostream>

using namespace std;

int main() {
  int n;
  int u, v, w;
  int c = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> u;
    cin >> v;
    cin >> w;
    if ((u + v + w) >= 2) {
      c++;
    }
  }
  cout << c;
}
