#include <iostream>
using namespace std;

int main() {

  int n, k, s = 0;
  cin >> n >> k;

  int racers[n];
  for (int i = 0; i < n; i++) {
    cin >> racers[i];
  }
  for (int i = 0; i < n; i++) {
    {
      if (racers[i] >= racers[k - 1] && racers[i] > 0)
        s++;
    }
  }
  cout << s << "\n";
}
