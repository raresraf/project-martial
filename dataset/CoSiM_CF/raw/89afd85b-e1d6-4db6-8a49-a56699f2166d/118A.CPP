#include <stdio.h>
#include <string>

using namespace std;

const char vowels[] = {'a', 'e', 'i', 'o', 'u', 'y'};

bool itsVowel(char letter) {
  for (int i = 0; i < 6; i++)
    if (letter == vowels[i])
      return true;
  return false;
}

int main() {
  char input[110];
  scanf("%s", &input);
  for (int i = 0; input[i]; i++) {
    if (!itsVowel(tolower(input[i]))) {
      printf(".%c", tolower(input[i]));
    }
  }
  return 0;
}