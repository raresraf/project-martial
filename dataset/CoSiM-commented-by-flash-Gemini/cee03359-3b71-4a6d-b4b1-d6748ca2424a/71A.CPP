#include <iostream>
#include <string>

using namespace std;

int main() {
  int t;
  cin >> t;
  for (int j = 0; j < t; j++) {
    char word[100];
    cin >> word;
    int i;
    for (i = 0; word[i] != '\0'; i++)
      ;
    if (i > 10)
      cout << word[0] << i - 2 << word[i - 1] << endl;
    else
      cout << word << endl;
  }
  return 0;
}
