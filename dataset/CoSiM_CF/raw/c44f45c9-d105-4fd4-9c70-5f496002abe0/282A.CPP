#include <iostream>
using namespace std;

int main() {
  int x = 0;
  int s;
  cin >> s;
  for (int i = 0; i < s; i++) {
    string newstatement;
    cin >> newstatement;

    if (newstatement[1] == '+')
      x++;

    else
      x--;
  }

  cout << x << "\n";
}
