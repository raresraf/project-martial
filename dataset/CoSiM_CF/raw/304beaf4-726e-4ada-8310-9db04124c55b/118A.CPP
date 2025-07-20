#include <iostream>
#include <string>
using namespace std;

int main() {
  string kata;
  int a;

  cin >> kata;

  a = kata.length();

  for (int i = 0; i < a; i++) {
    if (kata[i] != 'A' && kata[i] != 'I' && kata[i] != 'U' && kata[i] != 'E' &&
        kata[i] != 'O' && kata[i] != 'Y' && kata[i] != 'a' && kata[i] != 'i' &&
        kata[i] != 'u' && kata[i] != 'e' && kata[i] != 'o' && kata[i] != 'y') {
      cout << ".";
      if ((kata[i] >= 'A') && (kata[i] <= 'Z')) {
        kata[i] = kata[i] - 'A' + 'a';
        cout << kata[i];
      } else {
        cout << kata[i];
      }
    }
  }
}
