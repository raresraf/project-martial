#include <iostream>

int main() {
  std::string a, b = "hello";
  int n = 0, m = 0;
  std::cin >> a;

  for (int i = 0; i < a.length(); i++) {
    if (a[i] == b[n]) {
      n++;
      m++;
    }

    if (m == 5) {
      break;
    }
  }

  if (m == 5) {
    std::cout << "YES";
  } else {
    std::cout << "NO";
  }

  return 0;
}

/*std::string s;           FIRST TRY
    std::cin >> s;
    bool a[5];
    int suma = 0;

    for(int i = 0; i < s.length(); i++){
        if(s[i] == 'h'){
            a[0] = true;
            continue;
        }
        else if(s[i] == 'e' && a[0] == 1){
            a[1] = true;
            continue;
        }
        else if(s[i] == 'l' && a[1] == 1){
            a[2] = true;
            continue;
        }
        else if(s[i] == 'l' && a[2] == 1){
            a[3] = true;
            continue;
        }
        else if(s[i] == 'o' && a[3] == 1){
            a[4] = true;
            break;
        }
        else{
            continue;
        }
    }


    if(a[4]){
        std::cout << "YES";
    }
    else{
        std::cout << "NO";
    }*/