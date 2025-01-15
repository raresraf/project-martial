#include <iostream>
using namespace std;
int main() {
  string input;
  cin >> input;
  char temp;
  int pos;
  for (int i = 0; i < input.length(); i += 2) {
    pos = i;
    for (int j = i; j < input.length(); j += 2) {
      if (input[j] < input[pos]) {
        pos = j;
      }
    }
    temp = input[i];
    input[i] = input[pos];
    input[pos] = temp;
  }
  cout << input << endl;
}
