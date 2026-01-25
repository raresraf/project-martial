#include <iostream>

using namespace std;

int main() {
  int stops = 0;
  int enter = 0;
  int exit = 0;
  int maximum = 0;
  int total = 0;
  cin >> stops;
  for (int i = 0; i < stops; i++) {
    cin >> exit;
    cin >> enter;
    total = total + enter - exit;
    if (maximum < total) {
      maximum = total;
    }
  }
  cout << maximum;
  return 0;
}
