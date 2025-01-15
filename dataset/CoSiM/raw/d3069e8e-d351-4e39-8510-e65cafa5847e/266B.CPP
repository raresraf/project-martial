#include <iostream>
using namespace std;
int main() {
  int n, t;
  char Temp;
  string Que;
  cin >> n >> t;
  cin >> Que;
  while (t > 0) {
    for (int i = 0; i + 1 < n; i++) {
      if (Que[i] == 'B' && Que[i + 1] == 'G') {
        Temp = Que[i];
        Que[i] = Que[i + 1];
        Que[i + 1] = Temp;
        i++;
      }
    }
    --t;
  }
  cout << Que;
  return 0;
}
