#include <iostream>

using namespace std;

int main() {
  char word[100];
  cin >> word;
  int i = 0, person = 0;
  while (word[i + 1] != '\0') {
    if (word[i] == word[i + 1]) {
      person++;
      if (person > 5)
        break;
    } else
      person = 0;
    i++;
  }
  if (person > 5)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}