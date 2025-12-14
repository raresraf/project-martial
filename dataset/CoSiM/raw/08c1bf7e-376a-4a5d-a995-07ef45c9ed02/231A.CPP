#include <iostream>

using namespace std;

int main() {
  int problem;
  int Petya = 0;
  int Vasya = 0;
  int Tonya = 0;
  int counter = 0;
  cin >> problem;
  for (int i = 0; i < problem; i++) {
    cin >> Petya;
    cin >> Vasya;
    cin >> Tonya;
    int temp = Petya + Vasya + Tonya;
    if (temp >= 2) {
      counter++;
    }
  }
  cout << counter;
  return 0;
}