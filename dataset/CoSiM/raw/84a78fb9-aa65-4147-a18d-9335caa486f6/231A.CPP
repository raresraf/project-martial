#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;
  int count = 0;
  for (int i = 1; i <= n; i++) {
    int petya, vasya, tonya;
    cin >> petya >> vasya >> tonya;
    if (petya + vasya + tonya >= 2) {
      count++;
    }
  }
  cout << count;
  return 0;
}