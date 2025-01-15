// CodeForces
// Problem 118A [String Task]

#include <iostream>
using namespace std;

int main() {
  string str;
  char vowels[] = {'a', 'e', 'i', 'o', 'u', 'y'};
  cin >> str;

  int strlength = 0;
  while (str[strlength] != '\0') {
    strlength++;
  }

  for (int i = 0; i < strlength; i++) {
    int voweltrue = 0;
    str[i] = tolower(str[i]);
    for (int k = 0; k < 6; k++) {
      if (str[i] == vowels[k]) {
        voweltrue++;
      }
    }
    if (voweltrue == 0) {
      cout << "." << str[i];
    }
  }
  return 0;
}