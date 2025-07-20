#include <iostream>
using namespace std;
int main() {
  int ten = 10;
  int words;
  cin >> words;
  do {
    string input;
    cin >> input;
    int size;
    size = input.size();
    if (size <= 10) {
      cout << input << endl;

    } else {
      cout << input[0] << size - 2 << input[size - 1] << endl;
    }
  } while (--words);
}
