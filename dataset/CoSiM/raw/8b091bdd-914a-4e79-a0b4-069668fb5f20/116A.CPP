#include <iostream>
using namespace std;
int main() {
  int ar[1000];
  int enter, exit, r = 0;
  int station;
  cin >> station;
  for (int i = 0; i < station; i++) {
    cin >> exit >> enter;
    r = (r + enter) - exit;
    ar[i] = r;
  }
  int temp = 0;
  for (int i = 0; i < station; i++) {

    if (ar[i] > temp) {
      temp = ar[i];
    } else
      continue;
  }
  cout << temp;
}
