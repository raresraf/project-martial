#include <iostream>
#include <string>

int main() {
  std::string a, b, aMale, bMale;
  std::cin >> a;
  std::cin.get();
  std::cin >> b;
  int i = 0;
  char c;

  while (a[i]) {
    c = a[i];
    aMale += tolower(c);
    i++;
  }
  i = 0;
  while (b[i]) {
    c = b[i];
    bMale += tolower(c);
    i++;
  }

  int dlugosc = aMale.length();

  if (aMale < bMale) {
    std::cout << -1;
  } else if (aMale > bMale) {
    std::cout << 1;
  } else if (aMale == bMale) {
    std::cout << 0;
  }

  /*for(i = 0; i < dlugosc; i++){
      if(aMale[i] == bMale[i]){
          if(i == dlugosc - 1){
              std::cout << 0;
          }
          else
          {
              continue;
          }
      }
      else
      {
          if(aMale[i] > bMale[i]){
              std::cout << 1;
          }
          else
          {
              std::cout << -1;
          }
      }
  }*/

  return 0;
}