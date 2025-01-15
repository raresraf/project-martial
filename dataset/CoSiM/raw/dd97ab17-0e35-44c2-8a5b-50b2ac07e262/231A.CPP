#include <iostream>

using namespace std;

int main() {
  int n;
  int petya, vasya, tonya;
  int res = 0;
  cin >> n;
  while (n--) {
    cin >> petya >> vasya >> tonya;
    if (petya + vasya + tonya >= 2) {
      res++;
    }
  }
  cout << res;
}
